import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import rankdata
from models.ConsistancyEncoder import MultiHeadAttentionLayer
from models.losses import ClusteringConsistencyLoss

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, act=torch.relu, bias=False, dropout=0.0, residual=False):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.act = act
        self.bn = nn.BatchNorm1d(out_features)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x, adj):
        # x: [B,N,Fin], adj: [B,N,N]
        support = torch.bmm(x, self.weight.unsqueeze(0).expand(x.size(0), -1, -1))  # [B,N,Fout]
        out = torch.bmm(adj, support)  # [B,N,Fout]
        if self.bias is not None:
            out = out + self.bias
        out = self.bn(out.view(-1, out.shape[-1])).view(out.shape)
        out = self.act(out)
        if self.residual:
            out = out + x
        out = self.dropout(out)
        return out

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


def spearman_from_bold_np(bold_np: np.ndarray) -> torch.Tensor:
    """
    Compute Spearman's rank correlation for BOLD signals.
    bold_np: (N_nodes, T_time), variables in rows.
    Returns: (N_nodes, N_nodes) torch.float32
    """
    # Step 1: rank-transform each ROI time series along time axis
    # rankdata returns ranks starting from 1, ties averaged
    ranked = np.apply_along_axis(rankdata, 1, bold_np)  # shape: (N_nodes, T_time)

    # Step 2: Pearson correlation on the ranked data
    C = np.corrcoef(ranked)

    # Step 3: numerical safety
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 1.0)

    return torch.from_numpy(C).float()


# ------------------------------------------------------------
# NodeAlignment: map each view to a shared M-node space (soft pool)
# ------------------------------------------------------------
class NodeAlignment(nn.Module):
    def __init__(self, in_dim, out_node_num, feat_dim, dropout=0.0, layer='GCN'):
        super().__init__()
        self.layer = layer
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.out_node_num = out_node_num

        if self.layer == 'GCN':
            self.feat_gc = GraphConvolution(in_dim, feat_dim)
            self.pool_gc = GraphConvolution(in_dim, out_node_num)
        elif self.layer == 'Attention':
            self.feat_attn = MultiHeadAttentionLayer(in_dim, feat_dim, 1, dropout)
            self.pool_attn = MultiHeadAttentionLayer(in_dim, out_node_num, 1, dropout)
        else:
            raise NotImplementedError

    def forward(self, h, adj):
        """
        h:   [B, N_in, in_dim]
        adj: [B, N_in, N_in]
        return:
            pooled_feat: [B, M, feat_dim]  (M = out_node_num)
            entropy_reg: scalar
        """
        if self.layer in ['GCN']:
            feat = self.feat_gc(h, adj)               # [B, N_in, feat_dim]
            assign_logits = self.pool_gc(h, adj)      # [B, N_in, M]
        elif self.layer == 'Attention':
            feat, _ = self.feat_attn(h, h, h)         # [B, N_in, feat_dim]
            assign_logits, _ = self.pool_attn(h, h, h)  # [B, N_in, M]
        else:
            raise NotImplementedError

        assign = torch.softmax(assign_logits, dim=-1)           # [B, N_in, M]
        pooled_feat = torch.matmul(assign.transpose(1, 2), feat) # [B, M, feat_dim]
        entropy = (torch.distributions.Categorical(logits=assign_logits).entropy()).mean()
        return pooled_feat, entropy


# ------------------------------------------------------------
# ConsistancyAE: cross-view masked-node recovery
# ------------------------------------------------------------
class ConsistancyAE(nn.Module):
    """
    Cross-view masked-node recovery in aligned (shared M-node) latent space.

    For each batch of K views:
      1) z_i = model_mae.forward_encoder(x_i, adj_i, ...) -> [B, N_i, D]
      2) y_i = align(z_i) -> [B, M, D]
      3) Sample ONE shared mask on M nodes (ids_keep, ids_mask)
      4) For each pair (i <- j), reconstruct y_i[masked] from y_j[kept]
         using this module's per-dataset decoder/proj_out (not model_mae's)
      5) Loss = mean_{i!=j} SmoothL1( pred_i|j[masked], y_i[masked] ) + 0.01 * entropy
    """
    def __init__(self, datasets, atlas_names, hidden_dim=128, out_node_num=50, dropout=0.3, consistancy='cc'):
        super().__init__()
        self.datasets = datasets
        self.hidden_dim = hidden_dim
        self.out_node_num = out_node_num

        # Per-dataset modules
        self.transforms = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        # self.proj_outs = nn.ModuleDict()
        self.graph_proj = nn.ModuleDict()

        for n in atlas_names:
            self.transforms[n] = NodeAlignment(
                in_dim=hidden_dim, out_node_num=out_node_num,
                feat_dim=hidden_dim, dropout=dropout
            )
            # Using default batch_first=False -> we permute in forward
            self.decoders[n] = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout),
                num_layers=2
            )
            # self.proj_outs[n] = nn.Linear(hidden_dim, hidden_dim)

            self.graph_proj[n] = nn.Linear(hidden_dim, hidden_dim)

        # reconstruction loss
        self.consistancy = consistancy
        self.loss_fn = nn.SmoothL1Loss()
        if consistancy == 'cc':
            self.consistancy_loss = ClusteringConsistencyLoss(
                                        d_model=hidden_dim,
                                        num_prototypes=16,       # e.g., 64 prototypes
                                        temperature=1.0,
                                        entropy_weight=0.0,      # or small, e.g. 0.01
                                        diversity_weight=0.0     # or small, e.g. 0.01
                                    )
        else:
            self.consistancy_loss = None

    @torch.no_grad()
    def _shared_random_mask(self, B, M, ratio, device):
        """Return shared ids_keep / ids_mask / ids_restore for M nodes."""
        len_keep = max(4, int(M * (1.0 - ratio)))
        noise = torch.rand(B, M, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = torch.sort(ids_shuffle[:, :len_keep], dim=1).values
        ids_mask = torch.sort(ids_shuffle[:, len_keep:], dim=1).values
        return ids_keep, ids_mask, ids_restore

    def _reconstruct_from_other_view(self, z_keep_src, ids_keep, ids_mask,
                                     target_full, target_name):
        """
        Reconstruct target_full (view i) using kept nodes from source view j,
        with the TARGET view's decoder/proj_out (so we map to target's space).

        z_keep_src  : [B, K, D] (kept aligned nodes from source)
        target_full : [B, M, D] (aligned target for supervision)
        target_name : str key to pick per-dataset decoder/proj_out
        """
        B, M, D = target_full.shape
        K = z_keep_src.shape[1]
        mask_len = M - K

        # Assemble sequence: [kept ; mask tokens], here use zeros as generic mask tokens.
        # (You can swap in a learnable mask token if you prefer.)
        mask_tokens = torch.zeros(B, mask_len, D, device=target_full.device, dtype=target_full.dtype)
        x_merged = torch.cat([z_keep_src, mask_tokens], dim=1)  # [B, M, D]

        # Reorder to original (kept+masked) positions
        index_all = torch.cat([ids_keep, ids_mask], dim=1)      # [B, M]
        index_all_sorted = torch.argsort(index_all, dim=1)      # [B, M]
        x_reordered = torch.gather(
            x_merged, dim=1, index=index_all_sorted.unsqueeze(-1).expand(-1, -1, D)
        )                                                       # [B, M, D]

        # Loss on masked positions only
        pred_masked = torch.gather(x_reordered, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
        tgt_masked  = torch.gather(target_full, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
        loss = self.loss_fn(pred_masked, tgt_masked)
        return loss

    def _reconstruct_from_same_view(self, z_keep_src, ids_keep, ids_mask,
                                     target_full, target_name):
        """
        Reconstruct target_full (view i) using kept nodes from source view j,
        with the TARGET view's decoder/proj_out (so we map to target's space).

        z_keep_src  : [B, K, D] (kept aligned nodes from source)
        target_full : [B, M, D] (aligned target for supervision)
        target_name : str key to pick per-dataset decoder/proj_out
        """
        B, M, D = target_full.shape
        K = z_keep_src.shape[1]
        mask_len = M - K

        # Assemble sequence: [kept ; mask tokens], here use zeros as generic mask tokens.
        # (You can swap in a learnable mask token if you prefer.)
        mask_tokens = torch.zeros(B, mask_len, D, device=target_full.device, dtype=target_full.dtype)
        x_merged = torch.cat([z_keep_src, mask_tokens], dim=1)  # [B, M, D]

        # Reorder to original (kept+masked) positions
        index_all = torch.cat([ids_keep, ids_mask], dim=1)      # [B, M]
        index_all_sorted = torch.argsort(index_all, dim=1)      # [B, M]
        x_reordered = torch.gather(
            x_merged, dim=1, index=index_all_sorted.unsqueeze(-1).expand(-1, -1, D)
        )                                                       # [B, M, D]

        # Run through TARGET's decoder (batch_first=False by default)
        dec = self.decoders[target_name]
        pred = dec(x_reordered.permute(1, 0, 2)).permute(1, 0, 2)  # [B, M, D]

        # Loss on masked positions only
        pred_masked = torch.gather(pred, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
        tgt_masked  = torch.gather(target_full, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
        loss = self.loss_fn(pred_masked, tgt_masked)
        return loss

    def forward(self, model, batched_views, atlas_names, metas, device, current_epoch=0):
        """
        model_mae : provides forward_encoder (to produce latent z_i) — we DO NOT reuse its decoder/proj_out
        hs        : list of raw node features per view [B, N_i, F]
        adjs      : list of adj per view [B, N_i, N_i]
        data_names: list[str] keys matching what you used in __init__
        """
        assert len(batched_views) == len(atlas_names)
        K = len(batched_views)
        B = len(batched_views[0])
        mask_ratio = min(0.4, 0.1 + 0.05 * current_epoch)

        # 1) Encode each view (latent per-node)
        # 2) Align to shared M nodes (per dataset/view)
        aligned_feats, hg_views, aval_atlas = [], [], []
        entropy_reg, self_rec, cnt = 0.0, 0.0, 0
        for gs, name in zip(batched_views, atlas_names):
            if gs[0] is None:
                continue

            x = torch.stack(gs).to(device)
            adjs = (x > 0.3).float()
            batched_adjs = (adjs + adjs.transpose(1, 2)) / 2

            # Uses your BrainGFM/MAE encoder; returns (graph_repr, node_repr) or similar.
            hg, z, node_emb = model(
                x,
                parc_type=name
            )  # expect [B, N_i, D]
            if z.dim() == 2:
                z = z.unsqueeze(1)
            hg_views.append(hg)

            M = z.size(1)
            ids_keep, ids_mask, ids_restore = self._shared_random_mask(B, M, mask_ratio, device)
            loss_self = self._reconstruct_from_same_view(z, ids_keep, ids_mask, node_emb, name)
            self_rec = self_rec + loss_self

            pooled, ent = self.transforms[name](z, batched_adjs)   # [B, M, D], scalar
            aligned_feats.append(pooled)
            aval_atlas.append(name)
            entropy_reg = entropy_reg + ent
            cnt += 1
        if cnt > 0:
            entropy_reg = entropy_reg / cnt


        total_rec = 0.0
        total_rec_graph = 0.0
        if self.consistancy in ['cc']:
            if len(aval_atlas) >= 2:
                total_rec = total_rec + self.consistancy_loss(torch.stack(aligned_feats, dim=0).to(device))
                for i in range(len(aval_atlas)):  # target view index
                    tgt_name = aval_atlas[i]
                    for j in range(len(aval_atlas)):  # source view index
                        if i != j:
                            loss_ij_graph = self.loss_fn(self.graph_proj[tgt_name](hg_views[i]),
                                                         self.graph_proj[aval_atlas[j]](hg_views[j]))
                            total_rec_graph = total_rec_graph + loss_ij_graph

        # 5) Final loss
        total_loss = self_rec + entropy_reg + total_rec
        return total_loss, total_rec, entropy_reg, self_rec
