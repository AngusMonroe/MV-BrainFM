import os
import argparse
import random
import numpy as np
import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from collections import OrderedDict

from models.BNFM import MVBNFM
from models.ConsistancyAE import ConsistancyAE
from data.data import BrainDataset
from data.multi_data import MultiViewAlignedDataset, multiview_collate, GroupedBatchSampler
from models.disease_names import data2disease, name2n_nodes
from models.attention import CoordEncoder


def set_seed(seed, device):
    # setting seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def view_model_param(MODEL_NAME, model):
    total_param = 0
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


class MultiViewDataset(Dataset):
    def __init__(self, *datasets):
        assert len(datasets) >= 1
        # all datasets must be aligned and same length
        n = len(datasets[0].all)
        assert all(len(d.all) == n for d in datasets), "All datasets must have same length"
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0].all)

    def __getitem__(self, idx):
        # returns a K-tuple: one item from each dataset
        return tuple(d.all[idx] for d in self.datasets)


class MultiViewCollate:
    """Apply each dataset's own collate to its slice of the batch."""
    def __init__(self, collate_fns):
        # collate_fns: list of callables or None; length K
        self.collate_fns = collate_fns

    def __call__(self, batch):
        # batch: list of tuples; len(batch)=B, each tuple length=K
        # Transpose to per-view: views[i] is list length B for view i
        views = list(zip(*batch))  # K tuples, each of length B
        collated_views = []
        for view_items, coll in zip(views, self.collate_fns):
            # Each collate expects a list; tuples are fine, but cast to list if needed
            items_list = list(view_items)
            if coll is None:
                collated_views.append(default_collate(items_list))
            else:
                collated_views.append(coll(items_list))
        # Return a tuple aligned with views: (view0_batch, view1_batch, ...)
        return tuple(collated_views)


class ExP():
    def __init__(self, args, datasets, nsub, pretrain_mode, device):
        super(ExP, self).__init__()
        self.batch_size = args.batch_size
        self.n_epochs = args.epochs
        self.lr = args.lr
        self.b1 = 0.5
        self.b2 = 0.99
        self.nSub = nsub
        self.device = device
        self.save_name = args.save_name

        self.pretrain_sequence = None
        if '+' not in pretrain_mode and '->' in pretrain_mode:
            self.pretrain_sequence = pretrain_mode.split('->')
            self.pretrain_mode = self.pretrain_sequence[0]
        else:
            self.pretrain_mode = pretrain_mode.lower()

        self.save_path = f'./exp_results/fmri/graph_mae_pretrain/{pretrain_mode}/'
        os.makedirs(self.save_path, exist_ok=True)

        self.hidden_dim = None
        self.model = None
        self.optimizer = None

        self.datasets = datasets
        self.atlas_names = name2n_nodes.keys()

    def get_embarc_graph_data(self, path):
        return np.load(path, allow_pickle=True)

    def init_model(self, args):
        self.hidden_dim = args.hidden_dim

        bias_module = CoordEncoder(self.hidden_dim, args.nhead)

        self.model = MVBNFM(
            ff_hidden_size=args.hidden_dim,
            num_self_att_layers=args.self_att_layers,
            dropout=args.dropout,
            num_GNN_layers=args.gnn_layers,
            nhead=args.nhead,
            hidden_dim=args.hidden_dim,
            rwse_steps=args.rwse_steps,
            moe_num_experts=args.moe_experts,
            bias_module=bias_module
        ).to(self.device)
        
        view_model_param('MV-BrainFM', self.model)

        self.con_encoder = ConsistancyAE(
            datasets=self.datasets,
            atlas_names=self.atlas_names,
            hidden_dim=self.hidden_dim,
            dropout=args.dropout
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.con_encoder.parameters()),
            lr=self.lr,
            betas=(self.b1, self.b2)
        )

    def train_consistency(self):
        dataset = MultiViewAlignedDataset(self.datasets, view_order=self.atlas_names)
        sampler = GroupedBatchSampler(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, seed=42)

        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,  # <-- ensures same-dataset-per-batch
            collate_fn=multiview_collate,  # keep view alignment, tolerate missing views
            num_workers=0,
            pin_memory=True
        )

        loss_epoch, loss_epoch_con, loss_epoch_ent, loss_epoch_con_g = [], [], [], []
        min_loss = 1e9
        for epoch in range(self.n_epochs):
            self.model.train()
            self.con_encoder.train()
            losses, con_losses, ent_losses, con_g_losses = [], [], [], []

            gs_tqdm = tqdm(dataloader, desc=f"Epoch {epoch}", unit='batch')

            for batched_views, batched_label, batched_meta in gs_tqdm:
                total_loss, rec_loss, ent_loss, rec_g_loss = self.con_encoder(
                    model=self.model,
                    batched_views=batched_views,
                    atlas_names=self.atlas_names,
                    metas=batched_meta,
                    device=self.device,
                    current_epoch=epoch
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                all_params = list(self.model.parameters()) + list(self.con_encoder.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
                self.optimizer.step()

                losses.append(float(total_loss))
                con_losses.append(float(rec_loss))
                ent_losses.append(float(ent_loss))
                con_g_losses.append(float(rec_g_loss))

                gs_tqdm.set_postfix({
                    'Total Loss': f'{float(total_loss):.6f}',
                    'Con Loss': f'{float(rec_loss):.6f}',
                    'Ent Loss': f'{float(ent_loss):.6f}',
                    'ConG Loss': f'{float(rec_g_loss):.6f}'
                })

            epoch_loss = np.mean(losses)
            rec_epoch_loss = np.mean(con_losses)
            ent_epoch_loss = np.mean(ent_losses)
            rec_g_epoch_loss = np.mean(con_g_losses)

            model_filename = self.save_name
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                torch.save(self.model.state_dict(), os.path.join(self.save_path, model_filename))
            loss_epoch.append(epoch_loss)
            loss_epoch_con.append(rec_epoch_loss)
            loss_epoch_ent.append(ent_epoch_loss)
            loss_epoch_con_g.append(rec_g_epoch_loss)

        print(f"\n=== Final Consistency Loss ===")
        for i, l in enumerate(loss_epoch):
            print(f"Epoch {i + 1}: Loss = {l:.6f}")

        del dataset
        del dataloader
        del gs_tqdm
        del sampler
        torch.cuda.empty_cache()

        return loss_epoch[-1]


# -----------------------------
# Args / Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--gpu", type=int, default=1)

    # training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--dropout", type=float, default=0.5)

    # model
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--ff_hidden_size", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--self_att_layers", type=int, default=4)
    p.add_argument("--gnn_layers", type=int, default=4)
    p.add_argument("--rwse_steps", type=int, default=5)
    p.add_argument("--moe_experts", type=int, default=1)

    # IO
    p.add_argument("--save_name", type=str, default="bnfm.pth")
    return p.parse_args()


def main():
    args = parse_args()

    data_names = ['hbn_schaefer200', 'hbn_craddock200', 'hbn_aal116',
                  'adni_schaefer100', 'adni_schaefer200', 'adni_schaefer500', 'adni_aal116',
                  'abide_schaefer100', 'abide_schaefer200', 'abide_schaefer500', 'abide_aal116',
                  'ppmi_schaefer100', 'ppmi_schaefer200', 'ppmi_schaefer500', 'ppmi_aal116',
                  'taowu_schaefer100', 'taowu_schaefer200', 'taowu_schaefer500', 'taowu_aal116',
                  'neurocon_schaefer100', 'neurocon_schaefer200', 'neurocon_schaefer500', 'neurocon_aal116',
                  'adhd_schaefer100', 'adhd_schaefer200', 'adhd_schaefer500',
                  'oasis_schaefer100', 'oasis_schaefer200', 'oasis_schaefer500',
                  'renji_schaefer100', 'renji_schaefer200', 'renji_schaefer500',
                  'smhc_schaefer100', 'smhc_schaefer200', 'smhc_schaefer500',
                  'mdd_aal116', 'mdd_craddock200',
                  'drug_schaefer100', 'drug_schaefer200', 'drug_schaefer500', 'drug_aal116']

    pretrain_mode = "gcl"

    device = gpu_setup(True, args.gpu)

    exp = ExP(args, datasets=data_names, nsub=1, pretrain_mode=pretrain_mode, device=device)

    set_seed(42, device)
    exp.init_model(args)

    _ = exp.train_consistency()

    print(f'\n=== Pre-training with {pretrain_mode.upper()} Done! ===')


if __name__ == "__main__":
    main()
