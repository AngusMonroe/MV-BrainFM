import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.disease_names import name2n_nodes
from data.data import get_3d_coor


def pairwise_dist(xyz):  # xyz: [N,3], in atlas space (e.g., MNI mm)
    diff = xyz[:, None, :] - xyz[None, :, :]          # [N,N,3]
    D = torch.linalg.norm(diff, dim=-1)           # [N,N]
    return D

class CoordEncoder(nn.Module):
    """
    Soft Gaussian / RBF gate per head as an *additive attention bias*, but supports MULTI-ATLAS.

    - __init__ precomputes and stores ROI-ROI distance matrices for ALL atlases.
    - forward(atlas_name) returns bias for that atlas: [H, N_atlas, N_atlas]

    Bias (per head h):
        b_h(i,j) = alpha_h * ( - (D_ij - mu_h)^2 / (2 * sigma_h^2) ) + beta_h

    Key design choice for multi-atlas:
      - mu and sigma are learned in *normalized distance units* (fractions of dmax),
        then scaled by each atlas's dmax at forward time. This makes the same heads
        correspond to similar *relative* distance regimes across atlases.
    """
    def __init__(self, d_model, n_heads):
        """
        Parameters
        ----------
        atlas2xyz : dict[str, array-like]
            Mapping from atlas name to ROI coordinates of shape [N, 3].
            Example: {"aal116": xyz_aal, "schaefer200": xyz_s200, ...}
        n_heads : int
            Number of attention heads.
        The remaining args are kept only to preserve the *same* init signature style.
        (n_rbf, d_hidden are unused in this Gaussian-gate version.)
        centers, sigma can be used as initialization hints.
        """
        super().__init__()
        atlas2xyz = {
            name: get_3d_coor(name) for name in name2n_nodes.keys()
        }

        if not isinstance(atlas2xyz, dict) or len(atlas2xyz) == 0:
            raise ValueError("atlas2xyz must be a non-empty dict: {atlas_name: [N,3] coordinates}")

        self.n_heads = int(n_heads)

        # Store atlas names in a stable order
        self.atlas_names = sorted(list(name2n_nodes.keys()))

        # Precompute and register distance matrices + per-atlas dmax
        # We register them as buffers with safe names.
        self._atlas_key2buf = {}
        for name in self.atlas_names:
            safe = self._safe_name(name)
            self._atlas_key2buf[name] = safe

            xyz = torch.as_tensor(atlas2xyz[name], dtype=torch.float32)  # [N,3]
            with torch.no_grad():
                D = pairwise_dist(xyz)  # [N,N] (expects you have pairwise_dist defined)
                dmax = float(D.max().item() + 1e-8)

            self.register_buffer(f"D__{safe}", D)
            self.register_buffer(f"dmax__{safe}", torch.tensor(dmax, dtype=torch.float32))

        # -------- Head parameters in normalized units --------
        # mu_frac in (0,1) via sigmoid; sigma_frac > 0 via softplus.
        # Then: mu = mu_frac * dmax_atlas, sigma = sigma_frac * dmax_atlas.

        # Initialize mu fractions:
        # - If centers provided: map to fraction w.r.t. a representative dmax (median across atlases).
        # - Else: spread heads uniformly in [0,1].
        dmaxs = torch.tensor([getattr(self, f"dmax__{self._atlas_key2buf[n]}").item() for n in self.atlas_names])
        dmax_ref = float(torch.median(dmaxs).item() + 1e-8)

        init_mu_frac = torch.linspace(0.05, 0.95, steps=self.n_heads)

        # store in logit space
        p = init_mu_frac.clamp(1e-4, 1 - 1e-4)
        self.mu_frac_logit = nn.Parameter(torch.log(p) - torch.log(1 - p))  # [H]

        # Initialize sigma fractions:
        # If sigma provided: map to fraction; else default ~ 1/6 of dmax.
        init_sigma_frac = torch.full((self.n_heads,), 1.0 / 6.0, dtype=torch.float32)

        # store sigma_frac in softplus-invert space
        self.sigma_frac_unconstrained = nn.Parameter(torch.log(torch.exp(init_sigma_frac) - 1.0))

        # Per-head scale & offset (helps make it "mask-like" if needed)
        self.log_alpha = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))

    def forward(self, atlas_name: str):
        """
        Parameters
        ----------
        atlas_name : str
            Name key used in atlas2xyz at init.

        Returns
        -------
        bias : torch.Tensor
            Shape [H, N, N] for that atlas.
        """
        if atlas_name not in self._atlas_key2buf:
            raise KeyError(f"Unknown atlas_name={atlas_name}. Available: {self.atlas_names}")

        safe = self._atlas_key2buf[atlas_name]
        D = getattr(self, f"D__{safe}")           # [N,N]
        dmax = getattr(self, f"dmax__{safe}")     # scalar tensor

        # Constrained params in normalized units
        mu_frac = torch.sigmoid(self.mu_frac_logit)                       # [H] in (0,1)
        sigma_frac = F.softplus(self.sigma_frac_unconstrained) + 1e-6     # [H] > 0
        alpha = F.softplus(self.log_alpha) + 1e-6                         # [H] > 0
        beta = self.beta                                                  # [H]

        # Scale to atlas distance range
        mu = (mu_frac * dmax).view(self.n_heads, 1, 1)                    # [H,1,1]
        sigma = (sigma_frac * dmax).view(self.n_heads, 1, 1)              # [H,1,1]

        # Compute bias
        Dh = D.unsqueeze(0)                                               # [1,N,N]
        gauss = -0.5 * ((Dh - mu) ** 2) / (sigma ** 2)                    # [H,N,N]
        bias = alpha.view(self.n_heads, 1, 1) * gauss + beta.view(self.n_heads, 1, 1)

        return bias.contiguous()

    @staticmethod
    def _safe_name(name: str) -> str:
        # Make a buffer-friendly name
        safe = []
        for ch in name:
            if ch.isalnum():
                safe.append(ch)
            else:
                safe.append("_")
        safe = "".join(safe)
        if len(safe) == 0:
            safe = "atlas"
        return safe


BIG_NEG = -1e8  # safer than -inf for fp16

class DistBiasedSelfAttention(nn.Module):
    """
    Self-attention gated by ROI-ROI distance bias.
    The bias is turned into a multiplicative gate in [0,1] that scales attention weights.
    """
    def __init__(self, d_model, n_heads, dropout=0.0, bias_module=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.W_gate = nn.Linear(d_model, d_model, bias=False)

        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

        # Coord -> per-head bias [H,N0,N0]  (N0 = #ROIs for the atlas)
        self.bias_module = CoordEncoder(d_model, n_heads) if bias_module is None else bias_module

        # Gate sharpness; learnable scale on bias before sigmoid
        self.gate_alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

        self.mask = None
        self.attn = None

    @staticmethod
    def _build_additive_mask(B, H, N, src_mask=None, key_padding_mask=None, is_causal=None, device=None, dtype=None):
        """
        Returns an additive mask (float) of shape [B,H,N,N] with large negatives where masked.
        """
        if device is None: device = torch.device("cpu")
        if dtype is None: dtype = torch.float32

        add = torch.zeros(B, H, N, N, device=device, dtype=dtype)

        # src_mask: bool => hard mask; float => additive
        if src_mask is not None:
            if src_mask.dim() == 2:        # [N,N]
                m = src_mask[None, None, :, :].expand(B, H, N, N)
            elif src_mask.dim() == 3:      # [B,N,N]
                m = src_mask[:, None, :, :].expand(B, H, N, N)
            elif src_mask.dim() == 4:      # [B,1/H,N,N] or [B,H,N,N]
                m = src_mask
                if m.size(1) == 1:
                    m = m.expand(B, H, N, N)
            else:
                raise ValueError(f"Unsupported src_mask shape {tuple(src_mask.shape)}")

            if m.dtype == torch.bool:
                add = add.masked_fill(m, BIG_NEG)
            else:
                add = add + m.to(dtype=dtype)

        # key padding mask (mask keys/columns)
        if key_padding_mask is not None:
            pad = key_padding_mask[:, None, None, :].expand(B, H, N, N)
            add = add.masked_fill(pad, BIG_NEG)

        # causal
        if is_causal:
            causal = torch.ones((N, N), dtype=torch.bool, device=device).triu(1)
            add = add.masked_fill(causal[None, None, :, :], BIG_NEG)

        return torch.nan_to_num(add, neginf=BIG_NEG)

    @staticmethod
    def _pad_gate_to_tokens(gate_BHNN, N_target):
        """
        If x has extra tokens (e.g., 1 parc token), pad the [B,H,N0,N0] gate to [B,H,N_target,N_target]
        by placing the original gate in the lower-right block and setting new rows/cols to 1.0.
        """
        B, H, N0, _ = gate_BHNN.shape
        if N0 == N_target:
            return gate_BHNN
        assert N_target > N0, "Target length must be >= base ROI count"
        num_extra = N_target - N0
        device, dtype = gate_BHNN.device, gate_BHNN.dtype

        out = torch.ones(B, H, N_target, N_target, device=device, dtype=dtype)
        out[:, :, num_extra:, num_extra:] = gate_BHNN  # put ROI-ROI block at bottom-right
        # interaction between special tokens and ROIs/specials stays 1.0 (no suppression)
        return out

    @staticmethod
    def _gather_bias_with_ids(rel_bias_BHNN, ids_keep):
        """
        Select a subset of ROI indices consistently on both axes.
        rel_bias_BHNN: [B,H,N0,N0], ids_keep: [B,n_keep]
        returns [B,H,n_keep,n_keep]
        """
        B, H, N0, _ = rel_bias_BHNN.shape
        assert ids_keep.dim() == 2 and ids_keep.size(0) == B
        idx_row = ids_keep.unsqueeze(1).unsqueeze(-1).expand(B, H, ids_keep.size(1), N0)   # [B,H,nk,N0]
        sel_rows = torch.gather(rel_bias_BHNN, dim=2, index=idx_row)                        # [B,H,nk,N0]

        idx_col = ids_keep.unsqueeze(1).unsqueeze(1).expand(B, H, ids_keep.size(1), ids_keep.size(1))  # [B,H,nk,nk]
        gate = torch.gather(sel_rows, dim=3, index=idx_col)                                              # [B,H,nk,nk]
        return gate

    def forward(self, x, name=None, src_mask=None, key_padding_mask=None, is_causal=None, ids_keep=None):
        """
        x: [B,N,d]  (N may include 1 special token at front per BrainGFM)
        """
        B, N, _ = x.shape

        # x_norm = self.attn_norm(x)

        # Projections
        q = self.Wq(x).view(B, N, self.n_heads, self.dk).transpose(1, 2)  # [B,H,N,dk]
        k = self.Wk(x).view(B, N, self.n_heads, self.dk).transpose(1, 2)  # [B,H,N,dk]
        v = self.Wv(x).view(B, N, self.n_heads, self.dk).transpose(1, 2)  # [B,H,N,dk]

        # Base bias from coords: [H,N0,N0] -> [B,H,N0,N0]
        if name:
            base_bias = self.bias_module(name).to(device=x.device, dtype=x.dtype)  # [H,N0,N0]
            rel_bias = base_bias.unsqueeze(0).expand(B, -1, -1, -1)                # [B,H,N0,N0]

            # Optional node subsampling
            if ids_keep is not None:
                # Safety check to avoid OOB gathers that trigger CUDA asserts
                assert ids_keep.max() < rel_bias.size(-1), \
                    f"indices out of bounds: max={ids_keep.max().item()} >= {rel_bias.size(-1)}"
                rel_bias = self._gather_bias_with_ids(rel_bias, ids_keep)  # [B,H,nk,nk]

            self.mask = rel_bias

        # Build additive mask (logits space) for src/padding/causal
        add_mask = self._build_additive_mask(
            B, self.n_heads, N,
            src_mask=src_mask, key_padding_mask=key_padding_mask, is_causal=is_causal,
            device=x.device, dtype=x.dtype
        )                                                               # [B,H,N,N]

        # Scaled dot-product logits
        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.dk ** 0.5)  # [B,H,N,N]
        logits = logits + add_mask                                        # masked with BIG_NEG
        if name:
            logits = logits + rel_bias

        # Softmax -> multiply by gate -> renormalize
        attn = torch.softmax(logits, dim=-1)                              # [B,H,N,N]
        self.attn = attn

        # Weighted sum
        out = torch.matmul(attn, v)                                       # [B,H,N,dk]
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)

        out = self.attn_norm(x + self.dropout(out * torch.sigmoid(self.W_gate(x))))
        
        # Gate projection
        out = self.ffn_norm(x + self.dropout(self.Wo(out)))
        return out


class RandomFourierPositionalEncoding(nn.Module):
    """
    Random Fourier Feature (RFF) positional embedding for 3D coordinates.

    Args:
        out_dim (int): output embedding dimension.
        n_frequencies (int): number of random Fourier bases (K).
                             The final embedding before projection is 2*K.
        scale (float): frequency scaling factor (σ).
                       Smaller σ → lower frequencies; larger σ → higher.
    """

    def __init__(self, d_model, name2n_nodes, dropout, n_frequencies=64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.n_frequencies = n_frequencies

        # Random projection matrix B: shape [K, 3]
        self.B = nn.Parameter(torch.randn(n_frequencies, 3), requires_grad=True)
        nn.init.kaiming_normal_(self.B)

        # Linear projection to output dimension
        self.mlp = nn.ModuleDict({
            atlas: nn.Linear(name2n_nodes[atlas], d_model)
            for atlas in name2n_nodes.keys()
        })
        self.proj = nn.Linear(2 * n_frequencies, d_model)

        self.xyz = {name: get_3d_coor(name) for name in name2n_nodes.keys()}

    def forward(self, x, name, ids_keep=None):
        bz, _, _, = x.shape
        x = self.mlp[name](x)

        xyz = torch.as_tensor(self.xyz[name], dtype=torch.float32).to(device=x.device)  # [N,3]

        # [N,3] → [N,K]
        proj = 2 * math.pi * (xyz @ self.B.T)
        # Compute sin / cos
        pe = self.proj(torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1))
        if ids_keep is not None:
            # ids_keep: LongTensor [B, n_keep]
            assert ids_keep.dim() == 2 and ids_keep.size(0) == bz, \
                f"ids_keep must be [B, n_keep], got {tuple(ids_keep.shape)}"
            n_keep = ids_keep.size(1)

            # Select nodes from features: [B, n_keep, F]
            pe = torch.gather(
                pe.expand(bz, -1, -1), dim=1,
                index=ids_keep.unsqueeze(-1).expand(-1, -1, x.size(-1))
            )

        emb = x + pe
        return self.dropout(emb)
