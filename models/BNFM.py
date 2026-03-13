import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from models.disease_names import disease_names, name2n_nodes
from models.attention import DistBiasedSelfAttention, RandomFourierPositionalEncoding


class DiseaseGraphClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim=128, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        if self.num_classes > 1:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:  # regression
            self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj, parc_type, disease_type, valid_num_nodes=None):
        # frozen encoder
        g, _, _ = self.encoder(x, adj, parc_type, disease_type, valid_num_nodes)
        return self.classifier(g) if self.num_classes > 1 else self.classifier(g).squeeze(-1)


class MVBNFM(nn.Module):
    def __init__(self, ff_hidden_size, num_self_att_layers, dropout,
                 num_GNN_layers, nhead, hidden_dim=128, rwse_steps=5,
                 max_nodes=256, moe_num_experts=4, bias_module=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rwse_steps = rwse_steps
        self.max_nodes = max_nodes

        # === feature projections & tokens ===
        self.projection_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(num_GNN_layers)])
        self.disease_proj = nn.Linear(768, self.hidden_dim)
        self.trans = nn.Linear(self.hidden_dim + self.rwse_steps, self.hidden_dim)

        self.parcellation_tokens = nn.ParameterDict({
            atlas: nn.Parameter(torch.randn(1, 1, self.hidden_dim), requires_grad=True)
            for atlas in name2n_nodes.keys()
        })
        self.atlas_encoder = RandomFourierPositionalEncoding(self.hidden_dim, name2n_nodes, dropout=dropout)

        disease_embed_dict = get_disease_embeddings()
        self.disease_embeddings = {}
        for k, v in disease_embed_dict.items():
            param = nn.Parameter(v.unsqueeze(0).unsqueeze(0), requires_grad=True)  # [1,1,768]
            self.disease_embeddings[k.lower()] = param
            self.register_parameter(f'disease_embedding_{k.lower()}', param)

        self.brainnet = nn.ModuleList([BrainNetAttentionLayer(
                                        d_model=hidden_dim,
                                        nhead=nhead,
                                        dim_feedforward=ff_hidden_size,
                                        n_layer=num_self_att_layers,
                                        dropout=dropout,
                                        num_experts=moe_num_experts,
                                        bias_module=bias_module
                                    ) for _ in range(num_GNN_layers)])

        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_GNN_layers)])

    def compute_rwse(self, adj, k):
        B, N, _ = adj.shape
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)
        rw = adj.clone()
        diag_features = []
        for _ in range(k):
            rw_diag = torch.diagonal(rw, dim1=1, dim2=2).unsqueeze(-1)  # [B,N,1]
            diag_features.append(rw_diag)
            rw = torch.bmm(rw, adj)
        return torch.cat(diag_features, dim=-1)  # [B,N,k]

    def expand_adj_block(self, adj, num_tokens=2):
        B, N, _ = adj.shape
        new_N = N + num_tokens
        new_adj = torch.zeros(B, new_N, new_N, device=adj.device)
        new_adj[:, num_tokens:, num_tokens:] = adj
        for i in range(num_tokens):
            new_adj[:, i, :] = 1
            new_adj[:, :, i] = 1
        new_adj = new_adj - torch.diag_embed(torch.diagonal(new_adj, dim1=1, dim2=2))
        return new_adj

    def forward(self, node_features, parc_type, ids_keep=None):
        """
        node_features: [B,N,F]
        Adj_block:     [B,N,N]
        parc_type:     str (atlas key)
        disease_type:  str (unused in this pass)
        valid_num_nodes: List[int]
        ids_keep: Optional LongTensor [B, n_keep]
        """
        B, N, F = node_features.shape

        # 1) atlas-specific identity encoding
        if '_' in parc_type:
            parc_type = parc_type.split('_')[0]
        node_emb = self.atlas_encoder(node_features, parc_type, ids_keep=ids_keep)

        h = node_emb
        for i in range(len(self.brainnet)):
            h = self.brainnet[i](
                h, parc_type,
                # src_key_padding_mask=padding_mask,
                is_causal=False,
                ids_keep=ids_keep
            )  # [B,N+1,H]

        g = h.sum(dim=1) / N

        return g, h, node_emb


class BrainNetAttentionLayer(nn.Module):
    """
    BrainNetCNN-style block with attention:
      E2E (x2): distance-gated self-attention over nodes
      E2N:      attention pooling from nodes -> single graph token (parc_token if provided)
      N2G/G2N:  inject graph token back to nodes (gated), akin to N2G path influencing nodes
      MLP:      MoE-FFN on nodes
    IO is identical to your previous BrainNetAttentionLayer:
      forward(src, name, src_mask=None, is_causal=None, src_key_padding_mask=None, ids_keep=None, parc_token=None)
      -> returns node features [B,N,D]
    """
    def __init__(self, d_model, nhead, dim_feedforward, n_layer, dropout=0.1, num_experts=4, bias_module=None):
        super().__init__()

        self.L = n_layer
        self.conv = nn.ModuleList([DistBiasedSelfAttention(d_model, n_heads=nhead, dropout=dropout, bias_module=bias_module) for i in range(n_layer)])
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for i in range(n_layer)])
        self.act = nn.LeakyReLU(0.33)
        self.drop_half = nn.Dropout(p=dropout)

        self.ffn = FastMoEFFN(d_model, dim_feedforward, num_experts, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

        # residual drop
        self.res_drop = nn.Dropout(dropout)

    def forward(self, src, name, src_mask=None, is_causal=None, src_key_padding_mask=None, ids_keep=None):
        """
        src: [B,N,d_model] node features
        name: atlas key for distance bias inside DistBiasedSelfAttention
        src_mask: [N,N] / [B,N,N] (bool or float)
        src_key_padding_mask: [B,N] bool, True for pad
        ids_keep: optional [B, n_keep] LongTensor (ROI subset)
        returns: updated node features [B,N,d_model]
        """
        x = src
        # print(x.device)
        # print(self.conv[0].device)

        for i in range(self.L):
            x = self.conv[i](
                x, name,
                src_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                ids_keep=ids_keep
            )
            # x = self.norm[i](x + self.res_drop(self.act(attn)))

        # ---- Node-wise FFN + norm (final) ----
        ff = self.ffn(x)
        # y, aux = self.ffn(x, pe)  # y: [B, N, d_model]
        # print(y.shape, aux["expert_fraction"], aux["load_balance_loss"])
        x = self.ffn_norm(x + self.res_drop(self.act(ff)))

        return x
