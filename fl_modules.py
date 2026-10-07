"""
fl_modules.py — the four algorithmic components of ACF-XAI-FL.

  MDGP   Modality-Deconfounded Gradient Projection      (Sec. 3.4, Eqs. 19-27)
  TSMC   Temporal Similarity Momentum Correction         (Sec. 3.5, Eqs. 28-35)
  CEA    Contrastive Explainability Alignment            (Sec. 3.6, Eqs. 36-40)
  FTPMB  Federated Tumour Prototype Memory Bank          (Sec. 3.7, Eqs. 41-42)
"""
from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn.functional as F


# =========================================================================== MDGP
def mdgp_alpha(cam: torch.Tensor, mode: str = "eq25") -> float:
    """Adaptive projection coefficient from Grad-CAM variance (Eqs. 25-26)."""
    var = cam.flatten(1).var(dim=1, unbiased=False).mean().item()
    if mode == "sigmoid":                       # Algorithm 1 form
        return 1.0 / (1.0 + math.exp(-var))
    return 1.0 - math.exp(-var)                 # Eq. (25)


def mdgp_project(g_task: list[torch.Tensor], g_mod: list[torch.Tensor], alpha: float):
    """
    g_MDGP = g_task - alpha * (g_task . g_mod / ||g_mod||^2) * g_mod      (Eq. 24)
    Returns projected gradients and the deconfounding regulariser value
    L_DC = (g_task . g_mod)^2  (Eq. 27, reported; normalised by ||g_mod||^2 for scale).
    """
    dot = sum((a * b).sum() for a, b in zip(g_task, g_mod) if a is not None and b is not None)
    nrm = sum((b * b).sum() for b in g_mod if b is not None) + 1e-12
    coef = alpha * dot / nrm
    out = []
    for a, b in zip(g_task, g_mod):
        if a is None:
            out.append(None)
        elif b is None:
            out.append(a)
        else:
            out.append(a - coef * b)
    l_dc = (dot ** 2 / nrm).item()
    return out, l_dc, (dot / (nrm.sqrt() * math.sqrt(sum((a * a).sum().item() for a in g_task if a is not None)) + 1e-12)).item()


# =========================================================================== TSMC
class TSMCServer:
    """
    Server aggregation.
      S_ij  = cos(delta_i, delta_j)                                   (Eq. 28)
      R_i   = 1/(N-1) sum_{j!=i} S_ij                                  (Eq. 29)
      R_i^t = gamma R_i^{t-1} + (1-gamma) R_i                          (Eq. 30)
      w_i   = n_i R_i^t / sum_k n_k R_k^t                              (Eq. 31)
      dG    = sum_i w_i delta_i                                        (Eq. 32)
      M^t   = mu M^{t-1} + dG                                          (Eq. 34)
      theta = theta + eta_s * M^t                                      (Eq. 35; delta_i = theta_i - theta_G)
    With use_tsmc=False this reduces to FedAvg (sample-size weights, no momentum, eta_s=1).
    """

    def __init__(self, num_clients: int, cfg: dict):
        self.K = num_clients
        self.gamma = cfg.get("tsmc_gamma", 0.9)
        self.mu = cfg.get("tsmc_momentum", 0.8)
        self.eta_s = cfg.get("server_lr", 0.01)
        self.enabled = cfg.get("use_tsmc", True)
        self.R = torch.full((num_clients,), 1.0 / num_clients)   # R_k^0 = 1/K
        self.M = None                                              # v^0 = 0
        self.last_weights = None

    def aggregate(self, deltas: list[dict], n_samples: list[int], client_ids: list[int]) -> dict:
        n = torch.tensor(n_samples, dtype=torch.float32)
        if not self.enabled:
            w = n / n.sum()
            self.last_weights = w.tolist()
            return {k: sum(w[i] * deltas[i][k].float() for i in range(len(deltas))) for k in deltas[0]}

        N = len(deltas)
        G = torch.zeros(N, N, dtype=torch.float64)                          # Gram matrix, key by key (no flat copy)
        for k in deltas[0]:
            for i in range(N):
                for j in range(i, N):
                    G[i, j] += (deltas[i][k].double() * deltas[j][k].double()).sum()
        G = G + G.triu(1).T
        nrm = G.diag().sqrt() + 1e-12
        S = (G / nrm[:, None] / nrm[None, :]).float()                       # Eq. 28
        R_now = (S.sum(1) - S.diag()) / max(N - 1, 1)                       # Eq. 29
        R_now = R_now.clamp(min=0.0) + 1e-6                                 # keep weights non-negative
        for j, cid in enumerate(client_ids):
            self.R[cid] = self.gamma * self.R[cid] + (1 - self.gamma) * R_now[j]   # Eq. 30
        Rsel = self.R[client_ids]
        w = n * Rsel / (n * Rsel).sum()                                     # Eq. 31
        self.last_weights = w.tolist()
        dG = {k: sum(w[i] * deltas[i][k].float() for i in range(N)) for k in deltas[0]}   # Eq. 32
        if self.M is None:
            self.M = {k: torch.zeros_like(v) for k, v in dG.items()}
        self.M = {k: self.mu * self.M[k] + dG[k] for k in dG}               # Eq. 34
        return {k: self.eta_s * self.M[k] for k in dG}                      # step applied in Eq. 35


# =========================================================================== CEA
class CEAState:
    """Global explanation prototype P_E (Eqs. 38-39) and negative queue Q (Algorithm 1)."""

    def __init__(self, cfg: dict):
        self.rho = cfg.get("cea_proto_momentum", 0.9)
        self.tau = cfg.get("cea_temperature", 0.07)
        self.queue = deque(maxlen=cfg.get("cea_queue_size", 256))
        self.P_E = None

    def server_update(self, client_mean_h: list[torch.Tensor], client_all_h: list[torch.Tensor]):
        z_bar = torch.stack(client_mean_h).mean(0)                                         # Eq. 39
        self.P_E = z_bar.clone() if self.P_E is None else self.rho * self.P_E + (1 - self.rho) * z_bar  # Eq. 38
        for h in client_all_h:                                                              # Q <- Q U {h_prev}
            for row in h:
                self.queue.append(row.detach().cpu())

    def negatives(self, device):
        if len(self.queue) == 0:
            return None
        return torch.stack(list(self.queue)).to(device)


def cea_loss(h: torch.Tensor, P_E: torch.Tensor | None, negatives: torch.Tensor | None, tau: float):
    """InfoNCE alignment of local explanation embeddings to the global prototype (Eq. 40)."""
    if P_E is None:
        return h.new_zeros(())
    h = F.normalize(h, dim=1)
    pos = (h @ F.normalize(P_E.to(h.device), dim=0)) / tau               # (B,)
    if negatives is None or len(negatives) == 0:
        return h.new_zeros(())
    neg = h @ F.normalize(negatives, dim=1).T / tau                       # (B, |Q|)
    logits = torch.cat([pos[:, None], neg], dim=1)
    return F.cross_entropy(logits, torch.zeros(len(h), dtype=torch.long, device=h.device))


# =========================================================================== FTPMB
class FTPMB:
    """Persistent global class-prototype memory bank with momentum update (Eqs. 41-42)."""

    def __init__(self, num_classes: int, dim: int, momentum: float = 0.9):
        self.C, self.dim, self.lam = num_classes, dim, momentum
        self.P = torch.zeros(num_classes, dim)
        self.initialised = torch.zeros(num_classes, dtype=torch.bool)

    def server_update(self, client_class_sums: list[torch.Tensor], client_class_counts: list[torch.Tensor]):
        sums = torch.stack(client_class_sums).sum(0)
        cnts = torch.stack(client_class_counts).sum(0)
        for c in range(self.C):
            if cnts[c] == 0:
                continue
            z_c = sums[c] / cnts[c]                                       # Eq. 41
            if not self.initialised[c]:
                self.P[c], self.initialised[c] = z_c, True
            else:
                self.P[c] = self.lam * self.P[c] + (1 - self.lam) * z_c   # Eq. 42

    def loss(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """L_proto = || z - p_y ||^2 (only for classes whose prototype exists)."""
        P = self.P.to(z.device)
        mask = self.initialised.to(z.device)[y]
        if mask.sum() == 0:
            return z.new_zeros(())
        return ((z[mask] - P[y[mask]]) ** 2).sum(1).mean()
