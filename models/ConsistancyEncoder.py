import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BrainGFM_Gprompt import GraphConvolution


class ConsistancyEncoder(nn.Module):
    def __init__(self, datasets, hidden_dim=128):
        super().__init__()
        self.datasets = datasets

        # get all data names that appears multiple times
        data_names = [d.split('_')[0] for d in self.datasets]
        multi_data_names = set([name for name in data_names if data_names.count(name) > 1])

        self.transforms = nn.ModuleDict()
        for d in datasets:
            for data_name in multi_data_names:
                if d.startswith(data_name):
                    self.transforms[d] = NodeAlignment(in_dim=hidden_dim, out_node_num=50, feat_dim=hidden_dim, dropout=0.3, layer='GCN')

    def forward(self, hs, adjs, data_names, bz):
        hidden_hs = []
        total_ent_losses = 0.0
        for h, adj, data_name in zip(hs, adjs, data_names):
            h, ent_loss = self.transforms[data_name](h, adj, bz)
            hidden_hs.append(h)
            total_ent_losses += ent_loss

        # total_con_loss = self.row_cosine_align(torch.stack(hidden_hs, dim=0))
        total_con_loss = self.row_consensus_l2(torch.stack(hidden_hs, dim=0))
        # total_con_loss = 0.0
        # for i in range(len(hidden_hs) - 1):
        #     for j in range(i + 1, len(hidden_hs)):
        #         total_con_loss += self.subj_contrastive_loss(hidden_hs[i], hidden_hs[j])

        return total_con_loss, total_ent_losses

    def row_cosine_align(self, M):  # [K,n,d]
        Mnorm = F.normalize(M, dim=-1)
        # pairwise cosine for each row i across K
        # einsum gives [K,K,n]
        sims = torch.einsum('knd,lnd->kln', Mnorm, Mnorm)
        K = M.shape[0]
        tril = torch.tril(torch.ones(K, K, device=M.device), diagonal=-1).bool()
        loss = (1 - sims[tril].view(-1, M.shape[1]).mean(dim=0)).sum()
        return loss / M.shape[1]

    def row_consensus_l2(self, hs):  # M: [K, n, d]
        mu = hs.mean(dim=0, keepdim=True)  # [1, n, d]
        return ((hs - mu) ** 2).sum() / hs.numel()

    def subj_contrastive_loss(self, repr, feat, tau=0.75):
        bz = repr.size(0)
        feat = feat.view(bz, -1)
        repr = repr.view(bz, -1)

        # compute similarity matrix between `repr` and `feat`
        sim_mat = self.compute_similarity(repr, feat)
        sim_mat = torch.exp(sim_mat / tau)

        pos_mask = torch.eye(feat.size(0), dtype=torch.bool, device=feat.device)
        pos_loss = sim_mat[pos_mask].sum()
        neg_loss = sim_mat[~pos_mask].sum()
        loss = -math.log(pos_loss / neg_loss + 1e-8)

        return loss

    def compute_similarity(self, h1, h2):

        # Compute dot product between each pair of vectors
        dot_products = torch.matmul(h1, h2.t())  # Shape: (num_vectors, num_vectors)

        # Compute the norms of each vector
        norm1 = torch.norm(h1, p=2, dim=1, keepdim=True)  # Shape: (num_vectors, 1)
        norm2 = torch.norm(h2, p=2, dim=1, keepdim=True)

        # Normalize the dot products to obtain cosine similarity
        cosine_similarity = dot_products / (norm1 * norm2 + 1e-8)  # Add a small epsilon to avoid division by zero

        return cosine_similarity


class NodeAlignment(nn.Module):
    def __init__(self, in_dim, out_node_num, feat_dim, dropout=0.0, layer='GCN'):
        super().__init__()
        self.layer = layer
        if self.layer == 'GCN':
            self.feat_gc = GraphConvolution(in_dim, feat_dim)
            self.pool_gc = GraphConvolution(in_dim, out_node_num)
        elif self.layer == 'Attention':
            self.feat_attn = MultiHeadAttentionLayer(in_dim, feat_dim, 1, dropout)
            self.pool_attn = MultiHeadAttentionLayer(in_dim, out_node_num, 1, dropout)
        else:
            raise NotImplementedError
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.out_node_num = out_node_num
        # self.entropy_loss = 0.0

    def forward(self, h, adj, bz):
        device = h.device

        if self.layer in ['GCN']:
            feat = self.feat_gc(h, adj)
            assign_tensor = self.pool_gc(h, adj)
        elif self.layer == 'Attention':
            h = h.reshape(bz, -1, self.in_dim)
            feat, _ = self.feat_attn(h, h, h)
            assign_tensor, _ = self.pool_attn(h, h, h)
        else:
            raise NotImplementedError

        assign_tensor = torch.nn.functional.softmax(assign_tensor.view(bz, -1, self.out_node_num), dim=-1)
        feat = torch.matmul(assign_tensor.permute(0, 2, 1), feat.reshape(bz, -1, self.feat_dim)).reshape(-1, self.feat_dim)
        loss = self.cal_entropy_loss(assign_tensor)

        scale = torch.sqrt(torch.FloatTensor([feat.shape[0]])).to(device)
        feat = torch.matmul(feat, feat.t()) / scale
        return feat, loss

    def cal_entropy_loss(self, attn):
        entropy = (torch.distributions.Categorical(logits=attn).entropy()).mean()
        assert not torch.isnan(entropy)
        # self.entropy_loss = entropy #+ attn.norm(p=2)
        return entropy


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_heads, dropout, no_params=False, learnable_q=False):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.no_params = no_params

        # d_model // h 仍然是要能整除，换个名字仍然意义不变
        assert hid_dim % n_heads == 0

        if not self.no_params:
            self.w_q = nn.Linear(in_dim, hid_dim)
            self.w_k = nn.Linear(in_dim, hid_dim)
            self.w_v = nn.Linear(in_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        # self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):

        scale = torch.sqrt(torch.FloatTensor([self.hid_dim // self.n_heads])).to(query.device)

        # Q,K,V计算与变形：
        bsz = query.shape[0]

        if not self.no_params:
            Q = self.w_q(query)
            K = self.w_k(key)
            V = self.w_v(value)
        else:
            Q = query
            K = key
            V = value

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale

        # 如果没有mask，就生成一个
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：
        attention = self.dropout(torch.softmax(energy, dim=-1))

        # 第三步，attention结果与V相乘

        x = torch.matmul(attention, V)

        # 最后将多头排列好，就是multi-head attention的结果了

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        x = self.fc(x)

        return x, attention.squeeze()