import torch.utils.data
import random
# random.seed(42)
from models.disease_names import name2n_nodes
from data.data import BrainDataset
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
import math, random, torch
import pandas as pd
import os
import csv


def split_dataset_atlas(name: str):
    """
    'adni_schaefer100' -> ('adni', 'schaefer100')
    """
    parts = name.split('_')
    if len(parts) < 2:
        raise ValueError(f"Invalid dataset name '{name}'. Expect '<prefix>_<atlas>'.")
    return parts[0], parts[1]


def align_two_views_from_mapping(fmri_views, dti_views, mapping_csv_path, allow_none=False):
    """
    Align two modality/view lists using an index mapping CSV and discard mismatches.

    Parameters
    ----------
    fmri_views : Sequence[Any]
        List/tuple of fmri items (graphs/tensors/etc.). Indexed by fmri dataset order.
    dti_views : Sequence[Any]
        List/tuple of dti items (graphs/tensors/etc.). Indexed by dti dataset order.
    mapping_csv_path : str | Path
        CSV with columns: pid, idx_in_fmri, idx_in_dti
    allow_none : bool
        If False (default), rows where either selected item is None are dropped.
        If True, keep rows even if one item is None.

    Returns
    -------
    aligned_fmri : list
        fmri items aligned to dti using the mapping; length == len(aligned_dti).
    aligned_dti : list
        dti items aligned to fmri using the mapping; length == len(aligned_fmri).
    kept_meta : list[dict]
        Metadata per kept row: {'pid': str, 'fmri_idx': int, 'dti_idx': int, 'kept': bool}
    stats : dict
        Summary counts: {'rows_total', 'rows_kept', 'rows_dropped_oob', 'rows_dropped_none', 'rows_dropped_dup'}
    """
    df = pd.read_csv(mapping_csv_path)

    required_cols = {"pid", "idx_in_fmri", "idx_in_dti"}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Mapping CSV must have columns {required_cols}, got {df.columns.tolist()}")

    # Normalize column names (support case-insensitive headers)
    colmap = {c.lower(): c for c in df.columns}
    pid_col = colmap["pid"]
    fi_col = colmap["idx_in_fmri"]
    di_col = colmap["idx_in_dti"]

    n_fmri = len(fmri_views)
    n_dti  = len(dti_views)

    aligned_fmri, aligned_dti, kept_meta = [], [], []
    used_fmri = set()
    used_dti  = set()

    rows_total = len(df)
    dropped_oob = 0   # out-of-bounds indices
    dropped_none = 0  # one of the items is None and allow_none=False
    dropped_dup  = 0  # duplicate reference to the same index in a modality

    for _, row in df.iterrows():
        pid = row[pid_col]
        try:
            i_f = int(row[fi_col])
            i_d = int(row[di_col])
        except Exception:
            # Non-integer indices → drop
            dropped_oob += 1
            kept_meta.append({'pid': pid, 'fmri_idx': row[fi_col], 'dti_idx': row[di_col], 'kept': False})
            continue

        # Check bounds
        if not (0 <= i_f < n_fmri) or not (0 <= i_d < n_dti):
            dropped_oob += 1
            kept_meta.append({'pid': pid, 'fmri_idx': i_f, 'dti_idx': i_d, 'kept': False})
            continue

        # Avoid duplicates (same index referenced multiple times)
        if i_f in used_fmri or i_d in used_dti:
            dropped_dup += 1
            kept_meta.append({'pid': pid, 'fmri_idx': i_f, 'dti_idx': i_d, 'kept': False})
            continue

        f_item = fmri_views[i_f]
        d_item = dti_views[i_d]

        if not allow_none and (f_item is None or d_item is None):
            dropped_none += 1
            kept_meta.append({'pid': pid, 'fmri_idx': i_f, 'dti_idx': i_d, 'kept': False})
            continue

        aligned_fmri.append(f_item)
        aligned_dti.append(d_item)
        kept_meta.append({'pid': pid, 'fmri_idx': i_f, 'dti_idx': i_d, 'kept': True})
        used_fmri.add(i_f)
        used_dti.add(i_d)

    stats = {
        'rows_total': rows_total,
        'rows_kept': sum(1 for m in kept_meta if m['kept']),
        'rows_dropped_oob': dropped_oob,
        'rows_dropped_none': dropped_none,
        'rows_dropped_dup': dropped_dup,
    }
    return aligned_fmri, aligned_dti, kept_meta, stats


class MultiViewAlignedDataset(Dataset):
    """
    Build a multi-view dataset across datasets like:
      ['adni_schaefer100', 'adni_aal116',
       'abide_schaefer100', 'abide_schaefer200', 'abide_schaefer500', 'abide_aal116',
       'ppmi_schaefer100', 'ppmi_schaefer200', 'ppmi_schaefer500', 'ppmi_aal116']

    For each data prefix (adni/abide/ppmi), we align samples by a sample_id and
    produce a view list ordered by `name2n_nodes` keys. Missing views are `None`.

    __getitem__(i) -> (views_list, label, meta_dict)
      - views_list: list length == len(view_order), each entry is either a DGL graph (or your tensor)
                    for that atlas view, or None if missing.
      - label: taken from any available view (we check consistency).
      - meta_dict: {'group': <prefix>, 'sample_id': <id>}
    """
    def __init__(self, names, view_order=None, augment=False, multimodal=False, modal='fmri'):
        super().__init__()
        self.names = list(names)
        self.view_order = list(view_order) if view_order is not None else list(name2n_nodes.keys())
        self.internal_test = ['adhd', 'abide', 'adni']

        # 1) group requested names by (prefix -> list of atlases)
        groups = defaultdict(list)  # prefix -> [atlas_name_string]
        for n in self.names:
            if multimodal: 
                if n + '_dti' not in names or 'dti' in n:
                    continue
            elif modal == 'fmri' and 'dti' in n:
                continue
            elif modal == 'dti' and 'dti' not in n:
                continue
            prefix, atlas = split_dataset_atlas(n)
            groups[prefix].append(atlas)

        # 2) load all BrainDatasets once (per (prefix, atlas))
        #    NOTE: we assume you have BrainDataset available
        loaded = {}  # (prefix, atlas) -> dataset_instance
        for prefix, atlases in groups.items():
            for atlas in atlases:
                key = (prefix, atlas)
                ds_name = f"{prefix}_{atlas}"
            
                if multimodal:
                    fmri_data = BrainDataset(ds_name, split_data=False, augment=augment).all
                    dti_data = BrainDataset(ds_name + '_dti', split_data=False, augment=augment).all
    
                    mapping_path = '/data2/jiaxing/meta_data/{}_fmri_dti_mapping.csv'.format(prefix.split('_')[0])
                    aligned_fmri, aligned_dti, _, _ = align_two_views_from_mapping(fmri_data, dti_data, mapping_path)
                    loaded[(prefix, atlas + '_fmri')] = aligned_fmri
                    loaded[(prefix, atlas + '_dti')] = aligned_dti
                else:
                    if prefix in self.internal_test:
                        # for internal testing, use fixed small subsets
                        loaded[key] = BrainDataset(ds_name, split_data=True, augment=augment).train[1]
                    else:
                        loaded[key] = BrainDataset(ds_name, split_data=False, augment=augment).all
                    # Expect .all to be an iterable/list of tuples (graph, label, ...)

        # 3) build alignment dict per prefix:
        #    map sample_id -> {'label': y, 'views': {atlas: graph}}
        aligned_by_prefix = defaultdict(lambda: defaultdict(lambda: {'label': None, 'views': {}}))

        for (prefix, atlas), data_list in loaded.items():
            for idx, item in enumerate(data_list):
                # item should be (graph, label, ...) — adapt if needed
                graph = item[0]
                label = item[1]

                entry = aligned_by_prefix[prefix][idx]
                # label consistency check (silent overwrite if None)
                if entry['label'] is None:
                    entry['label'] = label
                else:
                    # if labels disagree across atlases, you may assert or prefer one
                    # assert int(entry['label']) == int(label), f"Label mismatch for {prefix}/{sid}"
                    pass
                entry['views'][atlas] = graph

        # 4) flatten into a single list of samples across all prefixes
        self.samples = []   # list of tuples: (views_list_in_order, label, {'group':prefix, 'sample_id':sid})
        for prefix, id_dict in aligned_by_prefix.items():
            for sid, payload in id_dict.items():
                views = []
                for vname in self.view_order:
                    # use None for missing views
                    views.append(payload['views'].get(vname, None))
                self.samples.append((views, payload['label'], {'group': prefix, 'sample_id': sid}))

        # shuffle order if you want a default randomized dataset order (keep DataLoader shuffle=True too)
        # import random; random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class GroupedBatchSampler(Sampler):
    """
    Yields lists of indices where every batch comes from a single dataset group.
    Groups are interleaved in a shuffled order each epoch.
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True,
                 drop_last: bool = False, seed: int | None = None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        # Build index buckets: group -> [idx, ...]
        self.group_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, _, meta = dataset[idx]
            grp = meta['group']  # e.g., 'adni', 'abide', 'ppmi'
            self.group_to_indices[grp].append(idx)

        self.groups = list(self.group_to_indices.keys())

    def __iter__(self):
        g = random.Random(self.seed) if self.seed is not None else random

        # Shuffle within each group
        buckets = {}
        for grp, idxs in self.group_to_indices.items():
            idxs = idxs.copy()
            if self.shuffle:
                g.shuffle(idxs)
            # Split into batches
            batches = [idxs[i:i+self.batch_size] for i in range(0, len(idxs), self.batch_size)]
            if self.drop_last and batches and len(batches[-1]) < self.batch_size:
                batches.pop()
            buckets[grp] = batches

        # Build an agenda of batches across groups
        agenda = []
        for grp, blist in buckets.items():
            for b in blist:
                agenda.append((grp, b))

        if self.shuffle:
            g.shuffle(agenda)

        # Yield batches (index lists only)
        for _, batch in agenda:
            yield batch

    def __len__(self):
        total = 0
        for idxs in self.group_to_indices.values():
            n_batches = math.floor(len(idxs) / self.batch_size) if self.drop_last else math.ceil(len(idxs) / self.batch_size)
            total += n_batches
        return total


# --- an example collate that preserves view alignment & tolerates missing views (None) ---
def multiview_collate(batch):
    """
    batch: list of (views_list, label, meta)
    returns:
        views_per_atlas: list length = V; each item is a list of graphs-or-None of length B
        labels: Tensor[B]
        metas:  list of len B
    """
    V = len(batch[0][0])
    B = len(batch)
    views_per_atlas = [[] for _ in range(V)]
    labels = []
    metas = []
    for (views, label, meta) in batch:
        for v in range(V):
            views_per_atlas[v].append(views[v])  # may be None
        labels.append(label)
        metas.append(meta)
    labels = torch.as_tensor(labels)
    return views_per_atlas, labels, metas
