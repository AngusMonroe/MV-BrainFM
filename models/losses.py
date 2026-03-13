import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_view_info_nce_loss(z: torch.Tensor, temperature: float = 0.1, eps: float = 1e-8) -> torch.Tensor:
    """
    z: [V, B, K, D]
      V: #views
      B: batch size (subjects)
      K: #supernodes
      D: feature dim

    Multi-view InfoNCE:
    - Positives: embeddings of the same (subject, supernode) across different views.
    - Negatives: all other (subject, supernode) positions (any view), excluding self.
    """
    V, B, K, D = z.shape
    device = z.device

    # Flatten subject & supernode into a single "slot" index
    N = B * K   # number of slots per view
    # reshape to [V, N, D]
    z = z.reshape(V, N, D)

    # Reorder to [N, V, D] then flatten to [N*V, D]
    # index i = slot * V + view  (slot: 0..N-1, view: 0..V-1)
    z = z.permute(1, 0, 2).contiguous()      # [N, V, D]
    z_flat = z.reshape(N * V, D)             # [N*V, D]

    # Normalize for cosine similarity
    z_flat = F.normalize(z_flat, p=2, dim=-1)

    # Similarity matrix: [N*V, N*V]
    sim = torch.matmul(z_flat, z_flat.t()) / temperature

    # Build masks -------------------------------------------------------
    num = N * V
    # slot_ids: [0,0,...0, 1,1,...1, ..., N-1,...]
    slot_ids = torch.arange(N, device=device).repeat_interleave(V)  # [N*V]
    # view_ids (not strictly needed, but could be useful)
    # view_ids = torch.arange(V, device=device).repeat(N)           # [N*V]

    # Positive mask: same slot_id, different view (exclude self)
    slot_eq = slot_ids.unsqueeze(0) == slot_ids.unsqueeze(1)        # [num, num]
    eye = torch.eye(num, device=device, dtype=torch.bool)
    pos_mask = slot_eq & ~eye                                      # same slot, not same element

    # For safety: anchors with no positives (shouldn't happen if V>=2)
    has_pos = pos_mask.any(dim=1)

    # Compute log-softmax over all *non-self* entries -------------------
    # Mask out self in denominator
    denom_mask = ~eye

    # exp(sim) with masks
    exp_sim = torch.exp(sim)
    exp_sim = exp_sim * denom_mask.float()

    # Numerator: sum over positives
    pos_exp = exp_sim * pos_mask.float()
    pos_sum = pos_exp.sum(dim=1) + eps

    # Denominator: sum over all non-self
    denom_sum = exp_sim.sum(dim=1) + eps

    log_prob = torch.log(pos_sum) - torch.log(denom_sum)   # [num]
    # Only keep anchors that actually have positives
    log_prob = log_prob[has_pos]

    loss = -log_prob.mean()
    return loss


def l2_consistency_loss(z: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    z: [V, B, K, D]
    Enforces that for each (b, k), embeddings across views are consistent.
    Uses variance around the multi-view mean, which is equivalent (up to a constant)
    to averaging all pairwise squared distances across views.
    """
    # z_mean: [1, B, K, D]
    z_mean = z.mean(dim=0, keepdim=True)
    diff_sq = (z - z_mean) ** 2   # [V, B, K, D]
    if reduction == "mean":
        return diff_sq.mean()
    elif reduction == "sum":
        return diff_sq.sum()
    else:
        # no reduction: return per-view/per-node loss
        return diff_sq.mean(dim=-1)  # [V, B, K]


class ClusteringConsistencyLoss(nn.Module):
    """
    Clustering-based consistency for multi-view supernode embeddings.

    Input:
        z: [V, B, K, D]
           V: number of views
           B: batch size (subjects)
           K: number of supernodes
           D: feature dimension

    Output:
        scalar loss (tensor) suitable to add to overall training objective.

    Internals:
        - A projection to M prototypes: logits -> soft assignments p in R^M via softmax.
        - Consistency: for each (b, k), make view-wise assignment distributions p[v,b,k,:]
          close to their mean distribution over views (symmetric KL).
        - Optional:
            * entropy regularization (encourage confident assignments),
            * diversity regularization (encourage using all prototypes).
    """

    def __init__(
        self,
        d_model: int,
        num_prototypes: int,
        temperature: float = 1.0,
        entropy_weight: float = 0.0,
        diversity_weight: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        self.eps = eps

        # Simple linear projection to prototype logits
        self.proj = nn.Linear(d_model, num_prototypes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [V, B, K, D]
        returns: scalar loss
        """
        V, B, K, D = z.shape
        assert D == self.d_model, "Last dimension of z must match d_model"

        # 1) Compute prototype logits and soft assignments
        # logits: [V, B, K, M]
        logits = self.proj(z)

        # temperature scaling (optional; often useful)
        logits = logits / max(self.temperature, self.eps)

        # soft assignments p: [V, B, K, M]
        p = F.softmax(logits, dim=-1) + self.eps  # avoid log(0)
        p = p / p.sum(dim=-1, keepdim=True)       # renormalize for safety

        # 2) Consistency term: make view distributions for each (b, k) similar
        # mean over views: p_mean: [1, B, K, M]
        p_mean = p.mean(dim=0, keepdim=True)

        # KL(p[v] || p_mean) and KL(p_mean || p[v]) to make it symmetric
        # KL(p || q) = sum p * (log p - log q)
        log_p = torch.log(p)
        log_p_mean = torch.log(p_mean)

        # KL(p[v,b,k] || p_mean[b,k])
        kl_pm = (p * (log_p - log_p_mean)).sum(dim=-1)           # [V, B, K]
        # KL(p_mean[b,k] || p[v,b,k])
        kl_mp = (p_mean * (log_p_mean - log_p)).sum(dim=-1)      # [V, B, K]

        # symmetric KL over views, then average over V,B,K
        cons_loss = 0.5 * (kl_pm + kl_mp).mean()

        # 3) Optional: entropy regularization (encourage confident assignments)
        # entropy per (v,b,k): H = -sum p log p
        if self.entropy_weight != 0.0:
            entropy = -(p * log_p).sum(dim=-1).mean()  # scalar
            # We add entropy (to minimize it), so high entropy => larger loss
            cons_loss = cons_loss + self.entropy_weight * entropy

        # 4) Optional: diversity regularization (use all prototypes)
        # global marginal over prototypes: q[m] = average usage of prototype m
        if self.diversity_weight != 0.0:
            # q: [M]
            q = p.mean(dim=(0, 1, 2)) + self.eps
            q = q / q.sum()
            M = self.num_prototypes
            # KL(q || uniform) = sum q log(q / (1/M))
            # = sum q (log q + log M)
            diversity_loss = (q * (torch.log(q) + torch.log(torch.tensor(M, device=z.device, dtype=z.dtype)))).sum()
            cons_loss = cons_loss + self.diversity_weight * diversity_loss

        return cons_loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewFactorizationLoss(nn.Module):
    """
    Multi-view factorization loss for pooled embeddings.

    Input:
        z: [V, B, K, D]
           V: #views
           B: batch size (subjects)
           K: #supernodes
           D: feature dimension (shared + private parts concatenated)

    Assumes:
        - First `shared_dim` channels of D are shared across views.
        - Remaining D - shared_dim channels are view-specific (private).

    Loss components:
        1) Shared consistency:
           - Encourage z_shared[v,b,k,:] to agree across views
             for each (b,k). (L2 around the view-mean.)

        2) Orthogonality / decorrelation:
           - Encourage shared and private subspaces to be uncorrelated
             (cross-covariance ~ 0).

        3) Private regularization (optional):
           - Simple L2 penalty on private magnitude, so model uses private
             part only when needed.
    """

    def __init__(
        self,
        d_model: int,
        shared_dim: int,
        lambda_shared: float = 1.0,
        lambda_orth: float = 1.0,
        lambda_private: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert 0 < shared_dim <= d_model, "shared_dim must be in (0, d_model]"
        self.d_model = d_model
        self.shared_dim = shared_dim
        self.lambda_shared = lambda_shared
        self.lambda_orth = lambda_orth
        self.lambda_private = lambda_private
        self.eps = eps

    def _shared_consistency(self, z_shared: torch.Tensor) -> torch.Tensor:
        """
        z_shared: [V, B, K, Ds]
        Enforce that for each (b,k), the views agree.

        We use variance around the view-mean:
            L = E_{b,k} [ Var_v ( z_shared[v,b,k,:] ) ]
        which is equivalent (up to constants) to pairwise L2 across views.
        """
        # mean over views: [1, B, K, Ds]
        mean = z_shared.mean(dim=0, keepdim=True)
        diff_sq = (z_shared - mean) ** 2  # [V, B, K, Ds]
        return diff_sq.mean()

    def _orthogonality(self, z_shared: torch.Tensor, z_priv: torch.Tensor) -> torch.Tensor:
        """
        Encourage shared and private subspaces to be decorrelated.

        z_shared: [V, B, K, Ds]
        z_priv:   [V, B, K, Dp]

        We flatten over (V,B,K) and compute cross-covariance C_sp:
            C_sp = (S^T P) / (N - 1), N = V*B*K
        and minimize ||C_sp||_F^2.
        """
        V, B, K, Ds = z_shared.shape
        _, _, _, Dp = z_priv.shape
        N = V * B * K

        # [N, Ds], [N, Dp]
        S = z_shared.reshape(N, Ds)
        P = z_priv.reshape(N, Dp)

        # center
        S = S - S.mean(dim=0, keepdim=True)
        P = P - P.mean(dim=0, keepdim=True)

        # cross-covariance: [Ds, Dp]
        C_sp = (S.T @ P) / max(N - 1, 1)

        # squared Frobenius norm
        return (C_sp ** 2).mean()

    def _private_reg(self, z_priv: torch.Tensor) -> torch.Tensor:
        """
        Simple L2 regularization on private part.

        z_priv: [V, B, K, Dp]

        Interpretation:
            - If lambda_private > 0, this discourages large private components.
            - The model is then incentivized to put view-invariant information
              into shared, and only use private when necessary (e.g., lesions).
        """
        return (z_priv ** 2).mean()

    def forward(self, z: torch.Tensor, return_components: bool = False):
        """
        z: [V, B, K, D]
        return_components:
            If True, also returns a dict with each component loss.
        """
        V, B, K, D = z.shape
        assert D == self.d_model, "Last dim of z must match d_model"

        Ds = self.shared_dim
        Dp = D - Ds
        assert Dp >= 0, "shared_dim larger than D"

        # Split into shared + private along feature dimension
        z_shared = z[..., :Ds]         # [V, B, K, Ds]
        z_priv = z[..., Ds:] if Dp > 0 else None

        # 1) Shared consistency loss
        loss_shared = self._shared_consistency(z_shared) if self.lambda_shared != 0.0 else z.new_tensor(0.0)

        # 2) Orthogonality loss (only if private exists and lambda_orth>0)
        if z_priv is not None and Dp > 0 and self.lambda_orth != 0.0:
            loss_orth = self._orthogonality(z_shared, z_priv)
        else:
            loss_orth = z.new_tensor(0.0)

        # 3) Private regularization
        if z_priv is not None and Dp > 0 and self.lambda_private != 0.0:
            loss_priv = self._private_reg(z_priv)
        else:
            loss_priv = z.new_tensor(0.0)

        # Total
        total_loss = (
            self.lambda_shared * loss_shared
            + self.lambda_orth * loss_orth
            + self.lambda_private * loss_priv
        )

        if return_components:
            return total_loss, {
                "loss_shared": loss_shared.detach(),
                "loss_orth": loss_orth.detach(),
                "loss_private": loss_priv.detach(),
            }
        else:
            return total_loss


def orthogonal_loss(tensor1, tensor2):
    # Compute dot product between normalized tensors
    norm_tensor1 = F.normalize(tensor1, dim=-1)
    norm_tensor2 = F.normalize(tensor2, dim=-1)
    dot_product = torch.sum(norm_tensor1 * norm_tensor2, dim=-1)
    loss = torch.mean(torch.abs(dot_product))

    return loss
