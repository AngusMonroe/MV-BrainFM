import os
import csv
import sys
import time
import math
import random
import dgl
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from models.disease_names import data2n_class
import warnings
warnings.filterwarnings('ignore')

# project imports
from models.BNFM import MVBNFM, DiseaseGraphClassifier
from data.data import BrainDataset
from models.disease_names import data2disease, name2n_nodes

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

# -----------------------------
# Utils
# -----------------------------
def gpu_setup(use_gpu: bool, gpu_id: int):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        return torch.device("cuda")
    print('cuda not available, using CPU')
    sys.exit()
    return torch.device("cpu")


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor,
                    average: str = "macro", multi_auc: str = "ovr", is_regression: bool = False):
    """
    Multiclass-safe metrics.
    - average: 'macro' | 'micro' | 'weighted' (for P/R/F1 and multiclass AUC)
    - multi_auc: 'ovr' or 'ovo' (strategy for multiclass AUC)

    Returns:
      {
        acc, auc, f1, precision, recall,
        sensitivity, specificity,
        per_class: {
          precision, recall, f1, sensitivity, specificity
        },
        confusion_matrix
      }
    }
    """
    with torch.no_grad():
        if is_regression:
            # For regression, return MSE/MAE
            mse = nn.MSELoss()(logits.view(-1), labels.float().view(-1)).item()
            mae = nn.L1Loss()(logits.view(-1), labels.float().view(-1)).item()
            return dict(mae=mae, mse=mse)
        y_true = labels.detach().cpu().numpy()
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # [B, C]
        y_pred = logits.argmax(dim=-1).detach().cpu().numpy()
        n_classes = probs.shape[1]
        all_labels = list(range(n_classes))  # stable label set

        # Basic accuracy
        acc = accuracy_score(y_true, y_pred)

        # Precision/Recall/F1 (averaged)
        prec_avg, rec_avg, f1_avg, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=all_labels, average=average, zero_division=0
        )

        # Per-class P/R/F1
        prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=all_labels, average=None, zero_division=0
        )

        # AUC
        # - Binary: standard AUC on positive class probs
        # - Multiclass: OvR/OvO with macro/micro/weighted averaging
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_true, probs[:, 1])
            else:
                auc = roc_auc_score(
                    y_true, probs, multi_class=multi_auc, average=average, labels=all_labels
                )
        except ValueError:
            # Happens if a class is missing in y_true for this batch
            auc = float("nan")

        # Confusion matrix (C x C)
        cm = confusion_matrix(y_true, y_pred, labels=all_labels).astype(np.float64)

        tp = np.diag(cm)
        fn = cm.sum(axis=1) - tp
        fp = cm.sum(axis=0) - tp
        tn = cm.sum() - (tp + fp + fn)

        eps = 1e-8
        sensitivity_pc = tp / np.maximum(tp + fn, eps)  # == recall per class
        specificity_pc = tn / np.maximum(tn + fp, eps)

        # Averaged sensitivity/specificity
        if n_classes == 2:
            # Keep the "positive" class convention for binary while also reporting macro
            # Positive class assumed to be class "1"
            pos = 1 if n_classes > 1 else 0
            sensitivity = float(sensitivity_pc[pos])
            specificity = float(specificity_pc[pos])
        else:
            # Macro-average for multiclass
            sensitivity = float(np.mean(sensitivity_pc))
            specificity = float(np.mean(specificity_pc))

        return dict(
            acc=float(acc),
            auc=float(auc),
            f1=float(f1_avg),
            precision=float(prec_avg),
            recall=float(rec_avg),
            sensitivity=sensitivity,
            specificity=specificity,
            per_class=dict(
                precision=prec_pc.tolist(),
                recall=rec_pc.tolist(),
                f1=f1_pc.tolist(),
                sensitivity=sensitivity_pc.tolist(),
                specificity=specificity_pc.tolist(),
            ),
            confusion_matrix=cm.tolist(),
        )


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Per-epoch warmup + cosine decay (reduces LR during training)."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1, last_epoch=-1):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        out = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * step / self.warmup_steps
            else:
                t = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = (self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))) * base_lr
            out.append(lr)
        return out


# -----------------------------
# Training wrapper (one fold)
# -----------------------------
class FoldRunner:
    def __init__(self, args, device, pretrained_path=None):
        self.args = args
        self.device = device

        name = args.data_name.split('_')[0].lower()
        self.task = 'regression' if data2n_class[name] == -1 else 'classification'

        encoder = MVBNFM(
            ff_hidden_size=args.ff_hidden_size,
            num_self_att_layers=args.self_att_layers,
            dropout=args.dropout,
            num_GNN_layers=args.gnn_layers,
            nhead=args.nhead,
            hidden_dim=args.hidden_dim,
            rwse_steps=args.rwse_steps,
            moe_num_experts=args.moe_experts,
        ).to(device)

        self.model = DiseaseGraphClassifier(
            encoder=encoder,
            hidden_dim=args.hidden_dim,
            num_classes=data2n_class[name]
        ).to(device)

        if pretrained_path and os.path.isfile(pretrained_path):
            print(f">>> Loading pretrained encoder from: {pretrained_path}")
            state = torch.load(pretrained_path, map_location=device)

            # Load ONLY into the BNFM encoder
            missing, unexpected = self.model.encoder.load_state_dict(state, strict=False)
            print("Missing keys in encoder:", missing)
            print("Unexpected keys in encoder:", unexpected)
        elif pretrained_path:
            print(f">>> WARNING: Pretrained not found: {pretrained_path}")

        self.criterion = nn.CrossEntropyLoss().to(device) if data2n_class[name] > 1 else nn.BCEWithLogitsLoss().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))

        warmup = max(1, int(args.epochs * args.warmup_ratio))
        self.scheduler = WarmupCosineLR(self.optimizer, warmup_steps=warmup,
                                        total_steps=args.epochs, min_lr_ratio=args.min_lr_ratio)

        self.attn = None
        self.mask = None

    def _run_loader(self, dataset, N, loader, train: bool):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        logits_all, labels_all = [], []
        for gs, label in loader:
            F = gs.shape[-1]
            label = label.to(self.device).long()
            node_feat = gs.to(self.device).view(-1, N, F)

            # simple symmetric adjacency (placeholder)
            adj = (node_feat > 0.3).float()
            adj = 0.5 * (adj + adj.transpose(1, 2))

            logits = self.model(
                node_feat, adj,
                parc_type=dataset.name.split('_')[1],
                disease_type=data2disease[dataset.name.split('_')[0]]
            )
            if self.task == 'regression':
                label = label.float()
            loss = self.criterion(logits, label)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
            else:
                self.attn = self.model.encoder.brainnet[-1].conv[-1].attn
                self.mask = self.model.encoder.brainnet[-1].conv[-1].mask

            total_loss += float(loss.item())
            logits_all.append(logits.detach())
            labels_all.append(label.detach())

        if len(loader) == 0:
            return {'loss': float('nan')}, None

        logits_all = torch.cat(logits_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        metrics = compute_metrics(logits_all, labels_all, is_regression=(self.task == 'regression'))
        metrics['loss'] = total_loss / max(1, len(loader))
        return metrics, (labels_all.cpu(), logits_all.argmax(dim=-1).cpu())

    def run_fold(self, dataset, N, fold_index, display_index):
        train_loader = DataLoader(dataset.train[fold_index], batch_size=self.args.batch_size, shuffle=True,
                                  drop_last=False, collate_fn=dataset.collate)
        val_loader = DataLoader(dataset.val[fold_index], batch_size=self.args.batch_size, shuffle=False,
                                drop_last=False, collate_fn=dataset.collate)
        test_loader = DataLoader(dataset.test[fold_index], batch_size=self.args.batch_size*2, shuffle=False,
                                 drop_last=False, collate_fn=dataset.collate)

        # remove the last item after '/' to get the save_path
        pth = self.args.pretrained.rsplit('/', 1)[0]
        save_path = pth + '/fintuned_ckpt/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        best_val_acc = -1.0 if self.task == 'classification' else float('inf')
        best_test = None
        best_epoch = -1

        pbar = tqdm(range(1, self.args.epochs + 1), desc=f"Fold {fold_index}", leave=True)
        for epoch in pbar:
            # ---- Train one epoch ----
            train_metrics, _ = self._run_loader(dataset, N, train_loader, train=True)

            # ---- Eval ----
            val_metrics, _ = self._run_loader(dataset, N, val_loader, train=False)
            test_metrics, _ = self._run_loader(dataset, N, test_loader, train=False)

            # ---- LR schedule (reduces LR over time) ----
            self.scheduler.step()
            lr_curr = self.optimizer.param_groups[0]["lr"]

            # ---- Track best by validation accuracy ----
            if self.task == 'classification' and val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                best_test = test_metrics
                best_epoch = epoch
                torch.save(self.attn, save_path + 'attn_fold_{}.pth'.format(display_index))
                torch.save(self.mask, save_path + 'mask_fold_{}.pth'.format(display_index))
            elif self.task == 'regression' and val_metrics['mae'] < best_val_acc:
                best_val_acc = val_metrics['mae']
                best_test = test_metrics
                best_epoch = epoch
                torch.save(self.attn, save_path + 'attn_fold_{}.pth'.format(display_index))
                torch.save(self.mask, save_path + 'mask_fold_{}.pth'.format(display_index))

            # ---- TQDM display ----
            if self.task == 'classification':
                pbar.set_postfix({
                    "lr": f"{lr_curr:.2e}",
                    "TrainLoss": f"{train_metrics.get('loss', float('nan')):.4f}",
                    "ValAcc": f"{val_metrics['acc']:.4f}",
                    "TestAcc": f"{test_metrics['acc']:.4f}",
                })
            else:
                pbar.set_postfix({
                    "lr": f"{lr_curr:.2e}",
                    "TrainLoss": f"{train_metrics.get('loss', float('nan')):.4f}",
                    "ValMAE": f"{val_metrics['mae']:.4f}",
                    "TestMAE": f"{test_metrics['mae']:.4f}",
                })
        if self.task == 'classification':
            print(f">>> Best val acc: {best_val_acc:.4f}, test acc: {best_test['acc']:.4f}, epoch: {best_epoch}")
        else:
            print(f">>> Best val MAE: {best_val_acc:.4f}, test MAE: {best_test['mae']:.4f}, epoch: {best_epoch}")

        return best_test  # metrics at best val acc


# -----------------------------
# Args / Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # data / CV
    p.add_argument("--data_name", type=str, default="HCPGender_schaefer100")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--fold", type=int, default=5)

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.5)

    # model
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--ff_hidden_size", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--self_att_layers", type=int, default=4)
    p.add_argument("--gnn_layers", type=int, default=4)
    p.add_argument("--rwse_steps", type=int, default=5)
    p.add_argument("--moe_experts", type=int, default=1)

    # LR schedule
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)

    # IO
    p.add_argument("--pretrained", type=str, default="./exp_results/fmri/graph_mae_pretrain/gcl/bnfm.pth")
    p.add_argument("--csv_path", type=str, default="./exp_results/fmri/graph_mae_pretrain/gcl/cv_summary_schaefer100.csv")
    return p.parse_args()


def main():
    args = parse_args()

    # Repro / device
    device = gpu_setup(True, args.gpu)
    set_seed(42, device)

    # Data
    dataset = BrainDataset(args.data_name, fold=args.fold)
    N = name2n_nodes[args.data_name.split('_')[1]]

    fold_metrics = []
    if args.data_name.split('_')[0].lower() in ['abide', 'adhd', 'adni']:
        seeds = [23, 42, 47, 233, 666, 888, 999, 1113, 1998, 2025]
        for i, seed in enumerate(seeds):
            set_seed(seed, device)
            runner = FoldRunner(args, device, pretrained_path=args.pretrained)
            best_test = runner.run_fold(dataset, N, 1, i)
            fold_metrics.append(best_test)
    else:
        # Run K folds
        for fold in range(args.fold):
            set_seed(42, device)
            runner = FoldRunner(args, device, pretrained_path=args.pretrained)
            best_test = runner.run_fold(dataset, N, fold, fold)
            fold_metrics.append(best_test)

    # Aggregate mean±std across folds
    keys = ["acc", "auc", "f1", "precision", "recall", "sensitivity", "specificity", "loss"] if runner.task == 'classification' else ["mae", "mse"]
    means = {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in keys}
    stds  = {k: float(np.nanstd([m[k] for m in fold_metrics])) for k in keys}

    # Prepare CSV header/row (one row per run)
    header = [
        "timestamp", "data_name",
        "lr", "warmup_ratio", "min_lr_ratio",
        "hidden_dim", "ff_hidden_size", "nhead", "self_att_layers", "gnn_layers",
        "rwse_steps", "moe_experts", "dropout",
        "pretrained",
        # metrics (mean±std)
        "acc", "auc", "f1", "precision", "recall", "sensitivity", "specificity", "loss"
    ]
    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"), args.data_name,
        args.lr, args.warmup_ratio, args.min_lr_ratio,
        args.hidden_dim, args.ff_hidden_size, args.nhead, args.self_att_layers, args.gnn_layers,
        args.rwse_steps, args.moe_experts, args.dropout,
        os.path.basename(args.pretrained) if args.pretrained else "",
        # format mean±std as a single string, as requested
        *(f"{means[k]:.6f}±{stds[k]:.6f}" for k in keys)
    ]

    # Write/append a single row
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
    write_header = not os.path.exists(args.csv_path)
    with open(args.csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    print("\n=== {}-fold Summary (mean ± std) ===".format(args.fold))
    for k in keys:
        print(f"{k}: {means[k]*100:.2f} ± {stds[k]*100:.2f}" if k != "loss" else f"{k}: {means[k]:.4f} ± {stds[k]:.4f}")
    print(f"\nSaved summary row to: {args.csv_path}")


if __name__ == "__main__":
    main()
