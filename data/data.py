import torch
import torch.utils.data
from torch.nn import functional as F
import time
import os
import numpy as np
import networkx as nx
import csv
import dgl
from models.disease_names import name2n_nodes
from tqdm import tqdm
import random
# random.seed(42)
from sklearn.model_selection import StratifiedKFold, train_test_split

class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


        This function is called inside a function in TUsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    # print(non_self_edges_idx)
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g

name2path = {
    'abide_schaefer100': '/data2/jiaxing/brain_pt/abide_full_schaefer100.pt',
    'abide_schaefer200': '/data2/jiaxing/brain_pt/abide_full_schaefer200.pt',
    'abide_schaefer500': '/data2/jiaxing/brain_pt/abide_full_schaefer500.pt',
    'abide_schaefer1000': '/data2/jiaxing/brain_pt/abide_full_schaefer1000.pt',
    'abide_aal116': '/data2/jiaxing/brain_pt/abide_full_aal116.pt',
    'abide_basc122': '/data2/jiaxing/brain_pt/abide_full_basc122_pearson.pt',

    # 'adni_aal116': '/data2/jiaxing/brain_pt/adni_AAL116.pt',
    # 'adni_harvard48': '/data2/jiaxing/brain_pt/adni_harvard48.pt',
    # 'adni_kmeans100': '/data2/jiaxing/brain_pt/adni_kmeans100.pt',
    # 'adni_schaefer100': '/data2/jiaxing/brain_pt/adni_schaefer100.pt',
    # 'adni_ward100': '/data2/jiaxing/brain_pt/adni_ward100.pt',
    
    'adni_aal116': '/data2/jiaxing/brain_pt/adni_IDEA_aal116.pt',
    'adni_basc122': '/data2/jiaxing/brain_pt/adni_IDEA_basc122.pt',
    'adni_schaefer100': '/data2/jiaxing/brain_pt/adni_IDEA_schaefer100.pt',
    'adni_schaefer200': '/data2/jiaxing/brain_pt/adni_IDEA_schaefer200.pt',
    'adni_schaefer300': '/data2/jiaxing/brain_pt/adni_IDEA_schaefer300.pt',
    'adni_schaefer400': '/data2/jiaxing/brain_pt/adni_IDEA_schaefer400.pt',
    'adni_schaefer500': '/data2/jiaxing/brain_pt/adni_IDEA_schaefer500.pt',
    
    'adni_schaefer100_dti': '/data2/jiaxing/brain_pt/adni_IDEA_schaefer100_dti.pt',
    'adni_schaefer200_dti': '/data2/jiaxing/brain_pt/adni_IDEA_schaefer200_dti.pt',
    'adni_schaefer300_dti': '/data2/jiaxing/brain_pt/adni_IDEA_schaefer300_dti.pt',
    'adni_schaefer400_dti': '/data2/jiaxing/brain_pt/adni_IDEA_schaefer400_dti.pt',
    'adni_schaefer500_dti': '/data2/jiaxing/brain_pt/adni_IDEA_schaefer500_dti.pt',

    'neurocon_aal116': '/data2/jiaxing/brain_pt/neurocon_AAL116.pt',
    'neurocon_harvard48': '/data2/jiaxing/brain_pt/neurocon_harvard48.pt',
    'neurocon_kmeans100': '/data2/jiaxing/brain_pt/neurocon_kmeans100.pt',
    'neurocon_schaefer100': '/data2/jiaxing/brain_pt/neurocon_schaefer100.pt',
    'neurocon_schaefer200': '/data2/jiaxing/brain_pt/neurocon_schaefer200.pt',
    'neurocon_schaefer500': '/data2/jiaxing/brain_pt/neurocon_schaefer500.pt',
    'neurocon_schaefer1000': '/data2/jiaxing/brain_pt/neurocon_schaefer1000.pt',
    'neurocon_ward100': '/data2/jiaxing/brain_pt/neurocon_ward100.pt',

    'ppmi_aal116': '/data2/jiaxing/brain_pt/ppmi_AAL116.pt',
    'ppmi_harvard48': '/data2/jiaxing/brain_pt/ppmi_harvard48.pt',
    'ppmi_kmeans100': '/data2/jiaxing/brain_pt/ppmi_kmeans100.pt',
    'ppmi_schaefer100': '/data2/jiaxing/brain_pt/ppmi_schaefer100.pt',
    'ppmi_schaefer200': '/data2/jiaxing/brain_pt/ppmi_schaefer200.pt',
    'ppmi_schaefer500': '/data2/jiaxing/brain_pt/ppmi_schaefer500.pt',
    'ppmi_schaefer1000': '/data2/jiaxing/brain_pt/ppmi_schaefer1000.pt',
    'ppmi_ward100': '/data2/jiaxing/brain_pt/ppmi_ward100.pt',

    'taowu_aal116': '/data2/jiaxing/brain_pt/taowu_AAL116.pt',
    'taowu_harvard48': '/data2/jiaxing/brain_pt/taowu_harvard48.pt',
    'taowu_kmeans100': '/data2/jiaxing/brain_pt/taowu_kmeans100.pt',
    'taowu_schaefer100': '/data2/jiaxing/brain_pt/taowu_schaefer100.pt',
    'taowu_schaefer200': '/data2/jiaxing/brain_pt/taowu_schaefer200.pt',
    'taowu_schaefer500': '/data2/jiaxing/brain_pt/taowu_schaefer500.pt',
    'taowu_schaefer1000': '/data2/jiaxing/brain_pt/taowu_schaefer1000.pt',
    'taowu_ward100': '/data2/jiaxing/brain_pt/taowu_ward100.pt',

    'matai_aal116': '/data2/jiaxing/brain_pt/matai_AAL116.pt',
    'matai_harvard48': '/data2/jiaxing/brain_pt/matai_harvard48.pt',
    'matai_kmeans100': '/data2/jiaxing/brain_pt/matai_kmeans100.pt',
    'matai_schaefer100': '/data2/jiaxing/brain_pt/matai_schaefer100_pearson.pt',
    'matai_ward100': '/data2/jiaxing/brain_pt/matai_ward100.pt',

    'hcpgender_schaefer100': '/data2/jiaxing/brain_pt/HCPGender_schaefer100.pt',
    'hcpgender_schaefer200': '/data2/jiaxing/brain_pt/HCPGender_schaefer200.pt',
    'hcpgender_aal116': '/data2/jiaxing/brain_pt/HCPGender_aal116.pt',
    'hcpgender_basc122': '/data2/jiaxing/brain_pt/HCPGender_basc122.pt',

    'hcpage_schaefer100': '/data2/jiaxing/brain_pt/HCPAge_schaefer100.pt',
    'hcpage_schaefer200': '/data2/jiaxing/brain_pt/HCPAge_schaefer200.pt',
    'hcpage_aal116': '/data2/jiaxing/brain_pt/HCPAge_aal116.pt',
    'hcpage_basc122': '/data2/jiaxing/brain_pt/HCPAge_basc122.pt',

    'hcpwm_schaefer100': '/data2/jiaxing/brain_pt/HCPWM_schaefer100.pt',
    'hcpwm_schaefer200': '/data2/jiaxing/brain_pt/HCPWM_schaefer200.pt',
    'hcpwm_aal116': '/data2/jiaxing/brain_pt/HCPWM_aal116.pt',
    'hcpwm_basc122': '/data2/jiaxing/brain_pt/HCPWM_basc122.pt',

    'huashan_schaefer100': '/data2/jiaxing/brain_pt/huashan_schaefer100.pt',
    'huashan_schaefer200': '/data2/jiaxing/brain_pt/huashan_schaefer200.pt',
    'huashan_schaefer300': '/data2/jiaxing/brain_pt/huashan_schaefer300.pt',
    'huashan_schaefer400': '/data2/jiaxing/brain_pt/huashan_schaefer400.pt',
    'huashan_schaefer500': '/data2/jiaxing/brain_pt/huashan_schaefer500.pt',

    'huashan_schaefer100_dti': '/data2/jiaxing/brain_pt/huashan_schaefer100_dti.pt',
    'huashan_schaefer200_dti': '/data2/jiaxing/brain_pt/huashan_schaefer200_dti.pt',
    'huashan_schaefer300_dti': '/data2/jiaxing/brain_pt/huashan_schaefer300_dti.pt',
    'huashan_schaefer400_dti': '/data2/jiaxing/brain_pt/huashan_schaefer400_dti.pt',
    'huashan_schaefer500_dti': '/data2/jiaxing/brain_pt/huashan_schaefer500_dti.pt',
    'huashan_schaefer1000_dti': '/data2/jiaxing/brain_pt/huashan_schaefer1000_dti.pt',

    'adhd_schaefer100': '/data2/jiaxing/brain_pt/adhd_schaefer100.pt',
    'adhd_schaefer200': '/data2/jiaxing/brain_pt/adhd_schaefer200.pt',
    'adhd_schaefer300': '/data2/jiaxing/brain_pt/adhd_schaefer300.pt',
    'adhd_schaefer400': '/data2/jiaxing/brain_pt/adhd_schaefer400.pt',
    'adhd_schaefer500': '/data2/jiaxing/brain_pt/adhd_schaefer500.pt',

    'oasis_schaefer100': '/data2/jiaxing/brain_pt/oasis_schaefer100.pt',
    'oasis_schaefer200': '/data2/jiaxing/brain_pt/oasis_schaefer200.pt',
    'oasis_schaefer300': '/data2/jiaxing/brain_pt/oasis_schaefer300.pt',
    'oasis_schaefer400': '/data2/jiaxing/brain_pt/oasis_schaefer400.pt',
    'oasis_schaefer500': '/data2/jiaxing/brain_pt/oasis_schaefer500.pt',

    'renji_schaefer100': '/data2/jiaxing/brain_pt/renji_schaefer100.pt',
    'renji_schaefer200': '/data2/jiaxing/brain_pt/renji_schaefer200.pt',
    'renji_schaefer300': '/data2/jiaxing/brain_pt/renji_schaefer300.pt',
    'renji_schaefer400': '/data2/jiaxing/brain_pt/renji_schaefer400.pt',
    'renji_schaefer500': '/data2/jiaxing/brain_pt/renji_schaefer500.pt',

    'smhc_schaefer100': '/data2/jiaxing/brain_pt/smhc_schaefer100.pt',
    'smhc_schaefer200': '/data2/jiaxing/brain_pt/smhc_schaefer200.pt',
    'smhc_schaefer300': '/data2/jiaxing/brain_pt/smhc_schaefer300.pt',
    'smhc_schaefer400': '/data2/jiaxing/brain_pt/smhc_schaefer400.pt',
    'smhc_schaefer500': '/data2/jiaxing/brain_pt/smhc_schaefer500.pt',

    'mdd_aal116': '/data2/jiaxing/brain_pt/mdd_aal116.pt',
    'mdd_craddock200': '/data2/jiaxing/brain_pt/mdd_craddock200.pt',
    'mdd_dosenbach160': '/data2/jiaxing/brain_pt/mdd_dosenbach160.pt',
    'mdd_power264': '/data2/jiaxing/brain_pt/mdd_power264.pt',

    'hbn_schaefer200': '/data2/jiaxing/brain_pt/HBN_schaefer200.pt',
    'hbn_schaefer300': '/data2/jiaxing/brain_pt/HBN_schaefer300.pt',
    'hbn_schaefer400': '/data2/jiaxing/brain_pt/HBN_schaefer400.pt',
    'hbn_aal116': '/data2/jiaxing/brain_pt/HBN_aal116.pt',
    'hbn_craddock200': '/data2/jiaxing/brain_pt/HBN_craddock200.pt',

    'drug_aal116': '/data2/jiaxing/brain_pt/drug_aal116.pt',
    'drug_schaefer100': '/data2/jiaxing/brain_pt/drug_schaefer100.pt',
    'drug_schaefer200': '/data2/jiaxing/brain_pt/drug_schaefer200.pt',
    'drug_schaefer500': '/data2/jiaxing/brain_pt/drug_schaefer500.pt',

    'drug_aal116_dti': '/data2/jiaxing/brain_pt/drug_aal116_dti.pt',
}

name2coor_path = {
    'atlas_200regions_5mm': '/data2/jiaxing/brain_coordinate/coordinate_atlas200.csv',
    'atlas_200regions_8mm': '/data2/jiaxing/brain_coordinate/coordinate_atlas200.csv',
    'schaefer1000': '/data2/jiaxing/brain_coordinate/schaefer1000_coordinates.csv',
    'schaefer100': '/data2/jiaxing/brain_coordinate/schaefer100_coordinates.csv',
    'schaefer200': '/data2/jiaxing/brain_coordinate/schaefer200_coordinates.csv',
    'schaefer500': '/data2/jiaxing/brain_coordinate/schaefer500_coordinates.csv',
    'aal116': '/data2/jiaxing/brain_coordinate/aal116_coordinates.csv',
    'craddock200': '/data2/jiaxing/brain_coordinate/craddock200_coordinates.csv',
    'dosenbach160': '/data2/jiaxing/brain_coordinate/dosenbach160_coordinates.csv',
    'power264': '/data2/jiaxing/brain_coordinate/power264_coordinates.csv',
    'basc122': '/data2/jiaxing/brain_coordinate/basc122_coordinates.csv',
}


def get_3d_coor(data_name):
    # 1. find matching atlas name
    name = ''
    for n in name2coor_path.keys():
        if n in data_name:
            name = n
            break
    if not name:
        raise ValueError("No coordinate file found for dataset {}".format(data_name))

    path = name2coor_path[name]
    n_nodes = name2n_nodes[name]

    # 2. read CSV into list of rows (as strings)
    rows = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # skip completely empty rows
            if not any(cell.strip() for cell in row):
                continue
            rows.append(row)

    arr = np.array(rows, dtype=object)  # (n_rows, n_cols)
    n_rows, n_cols = arr.shape

    # 3. column logic
    if n_cols == 4:
        # remove first column (e.g., ROI index)
        arr = arr[:, 1:]
        n_cols = arr.shape[1]
    elif n_cols > 4 or n_cols < 3:
        raise ValueError(f"Unexpected number of columns ({n_cols}) in {path}.")
    # now n_cols should be 3

    # 4. row logic
    if n_rows == n_nodes:
        pass  # OK
    elif n_rows == n_nodes + 1:
        # remove first row (header)
        arr = arr[1:, :]
        n_rows = arr.shape[0]
    else:
        raise ValueError(f"Unexpected number of rows ({n_rows}) in {path}.")

    # 5. convert to float
    try:
        coor = arr.astype(float)
    except ValueError as e:
        raise ValueError(f"Non-numeric value encountered in coordinate file {path}: {e}")

    return coor


def corr_from_bold_np(bold_np: np.ndarray) -> torch.Tensor:
    """
    bold_np: (N_nodes, T_time), variables in rows (as np.corrcoef expects).
    Returns: (N_nodes, N_nodes) torch.float32
    """
    C = np.corrcoef(bold_np)
    # NaN guard (can happen if a segment has zero variance in some ROI)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 1.0)
    return torch.from_numpy(C).float()


class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, name, split_data=True, augment=False, fold=10):
        t0 = time.time()
        self.name = name.lower()

        d = torch.load(name2path[self.name])
        mats = d["signals"]
        labels = d["labels"]

        if 'adni' in name:
            for i in range(labels.size(0)):
                if labels[i] == 3:
                    labels[i] = 2

        self.node_num = mats[0].size(0)
        self.coor = torch.from_numpy(get_3d_coor(self.name))
        self.dist = torch.cdist(self.coor, self.coor, p=2)

        print("[!] Dataset: ", self.name)

        data = []
        error_case = []
        min_feat_dim = mats[0].shape[-1]
        for i in range(len(mats)):
            if split_data and labels[i] < 0:
                labels[i] = 1e4
                error_case.append(i)
            if len(((mats[i] != 0).sum(dim=-1) == 0).nonzero()) > 0:
                for j in range(len(mats[i])):
                    if (mats[i][j] == 0).all():
                        # use random value to fill the empty node feature
                        mats[i][j] = torch.randn_like(mats[i][j])
                # error_case.append(i)
            if mats[i].shape[-1] < min_feat_dim:
                min_feat_dim = mats[i].shape[-1]
        print(error_case)
        # G_dataset = [n for i, n in enumerate(G_dataset) if i not in error_case]

        # realign the labels to 0~num_classes-1
        unique_labels = sorted(set(labels.tolist()))
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        labels = torch.LongTensor([label_mapping[label] for label in labels.tolist()])

        for i in tqdm(range(len(mats))):
            assert mats[i].size(0) == self.node_num

            if 'dti' in name or name == 'abide_basc122' or name == 'HCPGender_aal116':
                C = np.nan_to_num(mats[i].numpy(), nan=0.0, posinf=0.0, neginf=0.0)
                np.fill_diagonal(C, 1.0)
                conn = torch.from_numpy(C).float()
            else:
                conn = corr_from_bold_np(mats[i].numpy())

                if augment and not split_data:
                    bold = mats[i]  # (N, T)

                    # Split time axis into 3 parts as evenly as possible
                    chunks = torch.chunk(bold, chunks=4, dim=1)
                    for seg in chunks:
                        # skip degenerate segments (very short)
                        if seg.shape[1] < 2:
                            continue
                        conn_seg = corr_from_bold_np(seg.numpy())
                        data.append([conn_seg, labels.tolist()[i]])
            data.append([conn.float(), labels.tolist()[i]])

        dataset = self.format_dataset(data)
        self.all = self.format_dataset(data, error_case)

        if split_data:
            # this function splits data into train/val/test and returns the indices
            self.all_idx = self.get_all_split_idx(dataset)
            for split in range(fold):
                self.all_idx['train'][split] = [i for i in self.all_idx['train'][split] if i not in error_case]
                self.all_idx['val'][split] = [i for i in self.all_idx['val'][split] if i not in error_case]
                self.all_idx['test'][split] = [i for i in self.all_idx['test'][split] if i not in error_case]


            self.train = [self.format_dataset([dataset[idx] for idx in self.all_idx['train'][split_num]]) for split_num in
                          range(fold)]
            self.val = [self.format_dataset([dataset[idx] for idx in self.all_idx['val'][split_num]]) for split_num in
                        range(fold)]
            self.test = [self.format_dataset([dataset[idx] for idx in self.all_idx['test'][split_num]]) for split_num in
                         range(fold)]

        print("Time taken: {:.4f}s".format(time.time() - t0))

    def get_all_split_idx(self, dataset):
        """
            - Split total number of graphs into 3 (train, val and test) in 80:10:10
            - Stratified split proportionate to original distribution of data with respect to classes
            - Using sklearn to perform the split and then save the indexes
            - Preparing 10 such combinations of indexes split to be used in Graph NNs
            - As with KFold, each of the 10 fold have unique test set.
        """
        if 'dti' in self.name:
            root_idx_dir = './data/{}/'.format(self.name.split('_')[0] + '_dti')
        else:
            root_idx_dir = './data/{}/'.format(self.name.split('_')[0])
        if not os.path.exists(root_idx_dir):
            os.makedirs(root_idx_dir)
        all_idx = {}

        # If there are no idx files, do the split and store the files
        if not (os.path.exists(root_idx_dir + 'train.index')):
            print("[!] Splitting the data into train/val/test ...")

            k_splits = 10
            cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)
            k_data_splits = []

            # this is a temporary index assignment, to be used below for val splitting
            for i in range(len(dataset.graph_lists)):
                dataset[i][0].a = lambda: None
                setattr(dataset[i][0].a, 'index', i)

            for indexes in cross_val_fold.split(dataset.graph_lists, dataset.graph_labels):
                remain_index, test_index = indexes[0], indexes[1]

                remain_set = self.format_dataset([dataset[index] for index in remain_index])

                # Gets final 'train' and 'val'
                train, val, _, __ = train_test_split(remain_set,
                                                     range(len(remain_set.graph_lists)),
                                                     test_size=0.11,
                                                     stratify=remain_set.graph_labels)

                train, val = self.format_dataset(train), self.format_dataset(val)
                test = self.format_dataset([dataset[index] for index in test_index])

                # Extracting only idx
                idx_train = [item[0].a.index for item in train]
                idx_val = [item[0].a.index for item in val]
                idx_test = [item[0].a.index for item in test]

                f_train_w = csv.writer(open(root_idx_dir + 'train.index', 'a+'))
                f_val_w = csv.writer(open(root_idx_dir + 'val.index', 'a+'))
                f_test_w = csv.writer(open(root_idx_dir + 'test.index', 'a+'))

                f_train_w.writerow(idx_train)
                f_val_w.writerow(idx_val)
                f_test_w.writerow(idx_test)

            print("[!] Splitting done!")

        # reading idx from the files
        for section in ['train', 'val', 'test']:
            with open(root_idx_dir + section + '.index', 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, idx)) for idx in reader]
        return all_idx

    def format_dataset(self, dataset, error_case=[]):
        """
            Utility function to recover data,
            INTO-> dgl/pytorch compatible format
        """
        graphs = [data[0] for i, data in enumerate(dataset) if i not in error_case]
        labels = [data[1] for i, data in enumerate(dataset) if i not in error_case]

        return DGLFormDataset(graphs, labels)

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.from_numpy(np.array(labels))
        batched_graph = torch.cat(graphs, dim=0)

        return batched_graph, labels

    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim=0)  # .squeeze()
        deg_inv = torch.where(deg > 0, 1. / torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))

    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True
        for split_num in range(10):
            self.train[split_num].graph_lists = [self_loop(g) for g in self.train[split_num].graph_lists]
            self.val[split_num].graph_lists = [self_loop(g) for g in self.val[split_num].graph_lists]
            self.test[split_num].graph_lists = [self_loop(g) for g in self.test[split_num].graph_lists]

        for split_num in range(10):
            self.train[split_num] = DGLFormDataset(self.train[split_num].graph_lists,
                                                   self.train[split_num].graph_labels)
            self.val[split_num] = DGLFormDataset(self.val[split_num].graph_lists, self.val[split_num].graph_labels)
            self.test[split_num] = DGLFormDataset(self.test[split_num].graph_lists, self.test[split_num].graph_labels)
