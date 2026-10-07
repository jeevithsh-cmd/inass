"""
noise_config.py — round-level Gaussian perturbation of the client model update.

Implements Eq. (44) / Algorithm 1:  w_k <- w_k + N(0, sigma^2 C^2 I)
applied ONCE per communication round to each client's update before it is sent
to the server (T = 100 perturbation events per client over training).

C = 1.0, so sigma^2 C^2 I == sigma^2 I numerically (Algorithm 1). The clipping
threshold of 1.0 quoted in the paper is the LOCAL gradient clipping applied at
every optimisation step (cfg["grad_clip_norm"], see main_train.py). Clipping
the whole model update to norm C is available as an option (noise_clip_update)
but is not used for the paper's results.

IMPORTANT: this is NOT DP-SGD (no per-sample clipping / mini-batch-level noise)
and this module does NOT perform any privacy accounting. No (epsilon, delta)
budget is computed or claimed, consistent with the paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NoiseConfig:
    sigma: float = 0.01        # standard deviation (variance sigma^2 = 1e-4)
    clip_C: float = 1.0        # C: noise std = sigma * C
    clip_update: bool = False  # optionally clip the whole update to norm C first (off in the paper)
    enabled: bool = True

    @classmethod
    def from_cfg(cls, cfg: dict) -> "NoiseConfig":
        sigma = float(cfg.get("noise_sigma", 0.01))
        return cls(sigma=sigma,
                   clip_C=float(cfg.get("noise_clip_C", 1.0)),
                   clip_update=bool(cfg.get("noise_clip_update", False)),
                   enabled=bool(cfg.get("use_noise", True)) and sigma > 0)


def update_norm(delta: dict[str, torch.Tensor]) -> float:
    return torch.sqrt(sum((d.float() ** 2).sum() for d in delta.values())).item()


def clip_update(delta: dict[str, torch.Tensor], C: float) -> tuple[dict[str, torch.Tensor], float]:
    """Scale the update so that its global L2 norm is <= C."""
    norm = update_norm(delta)
    scale = min(1.0, C / (norm + 1e-12))
    return {k: v * scale for k, v in delta.items()}, norm


def perturb_update(delta: dict[str, torch.Tensor], nc: NoiseConfig,
                   generator: torch.Generator | None = None) -> tuple[dict[str, torch.Tensor], dict]:
    """Add N(0, sigma^2 C^2 I) to the client update. Returns (noisy_delta, info)."""
    if not nc.enabled:
        return delta, {"update_norm": None, "sigma": 0.0}
    if nc.clip_update:
        delta, norm = clip_update(delta, nc.clip_C)
    else:
        norm = update_norm(delta)
    out = {}
    for k, v in delta.items():
        noise = torch.randn(v.shape, generator=generator, device="cpu").to(v.device, v.dtype)
        out[k] = v + nc.sigma * nc.clip_C * noise
    return out, {"update_norm": norm, "sigma": nc.sigma, "C": nc.clip_C}
