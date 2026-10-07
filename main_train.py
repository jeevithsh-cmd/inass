"""
main_train.py — ACF-XAI-FL federated training and evaluation (Algorithm 1).

Usage
-----
  # full paper run (3 seeds: 42, 123, 456)
  python scripts/main_train.py --config configs/default.yaml

  # ablations / sensitivity (Tables 5, 7, 8) via overrides
  python scripts/main_train.py --config configs/default.yaml --set use_mdgp=false experiment_name=wo_mdgp
  python scripts/main_train.py --config configs/default.yaml --set heterogeneity_level=extreme experiment_name=het_extreme
  python scripts/main_train.py --config configs/default.yaml --set noise_sigma=0.05 experiment_name=sigma_0.05

  # centralised upper bound (used for "rounds to 85 % of centralised DSC", Table 6)
  python scripts/main_train.py --config configs/default.yaml --centralised --set experiment_name=centralised

  # smoke test on synthetic volumes (no BraTS data needed, CPU, ~1 min)
  python scripts/main_train.py --dry-run

Outputs (per seed) in outputs/<experiment_name>/seed_<s>/ :
  config_used.yaml, history.csv (validation curve, Figure 2), test_per_patient.csv,
  test_summary.json (Tables 3, 4, 9, 10), final_model.pt
"""
from __future__ import annotations

import argparse
import gc
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HOSPITALS, build_federated_datasets, build_synthetic_datasets  # noqa: E402
from fl_modules import CEAState, FTPMB, TSMCServer, cea_loss, mdgp_alpha, mdgp_project  # noqa: E402
from metrics import cam_iou, classification_metrics, seg_metrics  # noqa: E402
from models import ACFXAIFLNet  # noqa: E402
from noise_config import NoiseConfig, perturb_update  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("acf-xai-fl")
REPO = Path(__file__).resolve().parent.parent


# =========================================================================== config
def _parse_value(v: str):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        cfg[k] = _parse_value(v)
    return cfg


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================================================================== losses
def soft_dice_loss(logits, target, eps=1e-5):
    p = torch.sigmoid(logits)
    dims = tuple(range(2, p.ndim))
    inter = (p * target).sum(dims)
    den = p.sum(dims) + target.sum(dims)
    return (1 - (2 * inter + eps) / (den + eps)).mean()


def float_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items() if torch.is_floating_point(v)}


# =========================================================================== local training
def train_local(model, global_state, loader, cfg, cid, cea: CEAState, ftpmb: FTPMB, device):
    """E local epochs on one hospital. Returns update dict and statistics for the server."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    params = [p for p in model.parameters() if p.requires_grad]
    shared = model.shared_parameters()
    shared_ids = {id(p) for p in shared}
    psi_params = list(model.mod_head.parameters())
    psi_ids = {id(p) for p in psi_params}
    g_params = [p.detach().clone() for p in model.parameters()]          # w^{t-1} for proximal term
    negatives = cea.negatives(device) if cfg["use_cea"] else None

    stats = {"loss": [], "l_dc": [], "cos_task_mod": [], "alpha": []}
    h_rows, z_sum = [], torch.zeros(ftpmb.C, ftpmb.dim)
    z_cnt = torch.zeros(ftpmb.C)

    for epoch in range(cfg["local_epochs"]):
        for batch in loader:
            x = batch["image"].to(device)
            y_seg = batch["label"].to(device)
            y_cls = torch.as_tensor(batch["grade"], device=device).long()
            k = torch.full_like(y_cls, cid)

            out = model(x)
            l_ce = F.cross_entropy(out["cls_logits"], y_cls)
            l_dice = soft_dice_loss(out["seg_logits"], y_seg)

            # Grad-CAM explanation of the classification branch (Eq. 36) and embedding (Eq. 37)
            cam = model.gradcam(out["feat"], out["cls_logits"], target=y_cls, create_graph=cfg["use_cea"])
            h = model.explanation_embedding(cam if cfg["use_cea"] else cam.detach())
            l_cea = cea_loss(h, cea.P_E, negatives, cfg["cea_temperature"]) if cfg["use_cea"] else x.new_zeros(())
            l_proto = ftpmb.loss(out["z"], y_cls) if cfg["use_ftpmb"] else x.new_zeros(())
            l_prox = x.new_zeros(())
            if cfg["use_prox"]:
                l_prox = 0.5 * cfg["lambda_prox"] * sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), g_params))

            l_task = (l_ce + cfg["lambda_dice"] * l_dice + cfg["lambda_cea"] * l_cea
                      + cfg["lambda_proto"] * l_proto + l_prox)                       # Eq. 43 (task part)

            # auxiliary modality classifier psi (Eqs. 19-20)
            l_mod_full = F.cross_entropy(out["mod_logits"], k)                      # through encoder -> g_mod
            l_mod_psi = F.cross_entropy(model.mod_head(out["pooled"].detach()), k)   # trains psi only

            g_task = torch.autograd.grad(l_task, params, retain_graph=True, allow_unused=True)
            g_task = [g if id(p) not in psi_ids else None for g, p in zip(g_task, params)]

            if cfg["use_mdgp"]:
                g_mod = torch.autograd.grad(l_mod_full, shared, retain_graph=True, allow_unused=True)
                alpha = mdgp_alpha(cam.detach(), cfg.get("mdgp_alpha_mode", "eq25"))       # Eqs. 25-26
                idx = [i for i, p in enumerate(params) if id(p) in shared_ids]
                proj, l_dc, cosv = mdgp_project([g_task[i] for i in idx], list(g_mod), alpha)  # Eq. 24
                for j, i in enumerate(idx):
                    g_task[i] = proj[j]
                stats["l_dc"].append(cfg["lambda_dc"] * l_dc)                         # Eq. 27 (monitored)
                stats["cos_task_mod"].append(cosv)
                stats["alpha"].append(alpha)

            g_psi = torch.autograd.grad(l_mod_psi, psi_params, allow_unused=True)
            gp = dict(zip([id(p) for p in psi_params], g_psi))
            opt.zero_grad(set_to_none=True)
            for p, g in zip(params, g_task):
                p.grad = gp.get(id(p)) if id(p) in psi_ids else g
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            opt.step()
            stats["loss"].append(l_task.item())

            if epoch == cfg["local_epochs"] - 1:                                     # statistics for server
                h_rows.append(h.detach().cpu())
                zc = out["z"].detach().cpu()
                for c in range(ftpmb.C):
                    m = (y_cls.cpu() == c)
                    if m.any():
                        z_sum[c] += zc[m].sum(0)
                        z_cnt[c] += m.sum()

    local = float_state(model)
    delta = {n: local[n] - global_state[n] for n in global_state}                  # Delta w_k
    h_all = torch.cat(h_rows) if h_rows else torch.zeros(0, cfg["cea_embed_dim"])
    summary = {k: float(np.mean(v)) if v else None for k, v in stats.items()}
    return delta, h_all, z_sum, z_cnt, summary


# =========================================================================== evaluation


def evaluate(model, loader, device, with_cam=True):
    model.eval()
    rows, probs, ys = [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        y_seg = batch["label"].to(device)
        with torch.enable_grad():
            out = model(x)
            cam = model.gradcam(out["feat"], out["cls_logits"]) if with_cam else None
        r = seg_metrics(out["seg_logits"].detach(), y_seg)
        if with_cam:
            cam_up = F.interpolate(cam.detach(), size=x.shape[2:], mode="trilinear", align_corners=False)
            for a, b in zip(r, cam_iou(cam_up, y_seg)):
                a.update(b)
        p = torch.softmax(out["cls_logits"].detach(), 1).cpu().numpy()
        g = np.asarray(batch["grade"])
        for i, a in enumerate(r):
            a.update({"pid": batch["pid"][i], "grade": int(g[i]), "pred_grade": int(p[i].argmax())})
        rows += r
        probs.append(p)
        ys.append(g)
    return rows, np.concatenate(probs), np.concatenate(ys)


def summarise(rows):
    keys = [k for k in rows[0] if k not in ("pid", "grade", "pred_grade", "hospital")]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


# =========================================================================== federated loop
def run(cfg: dict, seed: int, dry_run: bool, centralised: bool):
    set_seed(seed, cfg.get("deterministic", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = REPO / cfg["output_dir"] / cfg["experiment_name"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump({**cfg, "seed": seed, "centralised": centralised}, f)

    data, preset = build_synthetic_datasets(cfg) if dry_run else build_federated_datasets(cfg)
    hosp = HOSPITALS[: cfg["num_clients"]]
    log.info("Seed %d | device %s | heterogeneity=%s | modalities=%s", seed, device,
             cfg.get("heterogeneity_level"), {h: preset[h] for h in hosp})

    g = torch.Generator()
    g.manual_seed(seed)
    mk = lambda ds, shuffle, bs: DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g if shuffle else None,
                                            num_workers=0 if dry_run else cfg.get("num_workers", 4))
    if centralised:     # one client holding all training data (upper bound)
        train_sets = [ConcatDataset([data[h]["train"] for h in hosp])]
        cfg = {**cfg, "use_tsmc": False, "use_noise": False, "use_mdgp": False}
    else:
        train_sets = [data[h]["train"] for h in hosp]
    loaders = [mk(ds, True, cfg["batch_size"]) for ds in train_sets]
    n_samples = [len(ds) for ds in train_sets]
    val_loaders = {h: mk(data[h]["val"], False, 1) for h in hosp}
    test_loaders = {h: mk(data[h]["test"], False, 1) for h in hosp}

    model = ACFXAIFLNet(cfg).to(device)
    gstate = float_state(model)
    server = TSMCServer(len(train_sets), cfg)
    cea = CEAState(cfg)
    ftpmb = FTPMB(cfg["num_grades"], cfg["ftpmb_embed_dim"], cfg["ftpmb_momentum"])
    nc = NoiseConfig.from_cfg(cfg)
    noise_gen = torch.Generator().manual_seed(seed + 1)
    log.info("Round-level perturbation: enabled=%s sigma=%.4f C=%.1f (no privacy accounting)",
             nc.enabled, nc.sigma, nc.clip_C)

    hist_f = open(out_dir / "history.csv", "w", newline="")
    hist = csv.writer(hist_f)
    hist.writerow(["round", "val_WT_DSC", "val_TC_DSC", "val_ET_DSC", "train_loss",
                   "tsmc_weights", "mdgp_alpha", "mdgp_cos_task_mod", "seconds"])

    for rnd in range(1, cfg["num_rounds"] + 1):
        t0 = time.time()
        deltas, h_means, h_alls, zs, zc, losses, alphas, coss = [], [], [], [], [], [], [], []
        for cid, loader in enumerate(loaders):
            model.load_state_dict({**model.state_dict(), **gstate})               # broadcast w^{t-1}
            delta, h_all, z_sum, z_cnt, st = train_local(model, gstate, loader, cfg, cid, cea, ftpmb, device)
            delta, _ = perturb_update(delta, nc, noise_gen)                       # Eq. 44, once per round
            deltas.append({k: v.cpu() for k, v in delta.items()})
            if len(h_all):
                h_means.append(h_all.mean(0))
                h_alls.append(h_all)
            zs.append(z_sum)
            zc.append(z_cnt)
            losses.append(st["loss"])
            alphas.append(st["alpha"])
            coss.append(st["cos_task_mod"])

        step = server.aggregate(deltas, n_samples, list(range(len(deltas))))      # TSMC (Eqs. 28-35)
        gstate = {k: gstate[k] + step[k].to(gstate[k].device) for k in gstate}
        del deltas, step
        gc.collect()
        if cfg["use_cea"] and h_means:
            cea.server_update(h_means, h_alls)                                    # Eqs. 38-39 + queue
        if cfg["use_ftpmb"]:
            ftpmb.server_update(zs, zc)                                           # Eqs. 41-42

        row_val = ["", "", ""]
        if rnd % cfg.get("eval_every", 5) == 0 or rnd == cfg["num_rounds"]:
            model.load_state_dict({**model.state_dict(), **gstate})
            vrows = []
            for h in hosp:
                vrows += evaluate(model, val_loaders[h], device, with_cam=False)[0]
            s = summarise(vrows)
            row_val = [f"{s['WT_DSC']:.4f}", f"{s['TC_DSC']:.4f}", f"{s['ET_DSC']:.4f}"]
            log.info("Round %3d/%d | val WT %.4f TC %.4f ET %.4f", rnd, cfg["num_rounds"],
                     s["WT_DSC"], s["TC_DSC"], s["ET_DSC"])
        fmt = lambda xs: ";".join("" if v is None else f"{v:.4f}" for v in xs)
        hist.writerow([rnd, *row_val, f"{np.mean(losses):.4f}", fmt(server.last_weights or []),
                       fmt(alphas), fmt(coss), f"{time.time() - t0:.1f}"])
        hist_f.flush()
    hist_f.close()

    # ---------------------------------------------------------------- final test (model at T)
    model.load_state_dict({**model.state_dict(), **gstate})
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    all_rows, per_h = [], {}
    all_p, all_y = [], []
    for h in hosp:
        rows, p, y = evaluate(model, test_loaders[h], device, with_cam=True)
        for r in rows:
            r["hospital"] = h
        per_h[h] = summarise(rows)
        all_rows += rows
        all_p.append(p)
        all_y.append(y)
    with open(out_dir / "test_per_patient.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    summary = {"seed": seed, "experiment": cfg["experiment_name"],
               "segmentation_and_xai": summarise(all_rows),                    # Tables 3 and 10
               "classification": classification_metrics(np.concatenate(all_y), np.concatenate(all_p)),  # Table 4
               "per_hospital": per_h}                                          # Table 9
    with open(out_dir / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Seed %d TEST | WT %.4f TC %.4f ET %.4f | Acc %.1f%% | saved to %s", seed,
             summary["segmentation_and_xai"]["WT_DSC"], summary["segmentation_and_xai"]["TC_DSC"],
             summary["segmentation_and_xai"]["ET_DSC"], summary["classification"]["Accuracy"], out_dir)
    return summary


def main():
    ap = argparse.ArgumentParser(description="ACF-XAI-FL training")
    ap.add_argument("--config", default=str(REPO / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="override config: key=value ...")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--centralised", action="store_true", help="single-client upper bound")
    ap.add_argument("--dry-run", action="store_true", help="synthetic data, 2 rounds, 1 epoch, 1 seed")
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    if a.seeds:
        cfg["seeds"] = a.seeds
    if a.dry_run:
        cfg.update(num_rounds=2, local_epochs=1, eval_every=1, seeds=cfg["seeds"][:1],
                   experiment_name="dry_run", batch_size=1)
        for kv in a.set:                                   # explicit overrides still win
            k, v = kv.split("=", 1)
            cfg[k] = _parse_value(v)
    results = [run(cfg, s, a.dry_run, a.centralised) for s in cfg["seeds"]]
    if len(results) > 1:
        wt = [r["segmentation_and_xai"]["WT_DSC"] for r in results]
        log.info("WT DSC over seeds %s: %.4f +- %.4f", cfg["seeds"], np.mean(wt), np.std(wt, ddof=1))


if __name__ == "__main__":
    main()
