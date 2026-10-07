"""
gradcam_xai.py — Grad-CAM saliency maps from the classification branch (Figure 4).

Grad-CAM is computed on the MMAM-recalibrated bottleneck F' w.r.t. the predicted
(or given) tumour-grade logit (Eq. 36), scaled to [0, 1], upsampled to the input
size and overlaid on the chosen modality. If a segmentation file is given, the
IoU of the CAM (threshold 0.5) with WT / TC / ET is printed (Table 10 protocol).

    python scripts/gradcam_xai.py \
        --checkpoint outputs/acf_xai_fl_full/seed_42/final_model.pt \
        --patient    data/BraTS2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001 \
        --hospital   D --output_dir outputs/xai_maps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HETEROGENEITY_PRESETS, MODALITIES, build_transforms, modality_mask  # noqa: E402
from metrics import cam_iou  # noqa: E402
from models import ACFXAIFLNet  # noqa: E402

GRADES = ["Grade II", "Grade III", "Grade IV"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="defaults to config_used.yaml next to the checkpoint")
    ap.add_argument("--patient", required=True, help="BraTS subject folder")
    ap.add_argument("--hospital", default="D", choices=list("ABCD"), help="modality availability to simulate")
    ap.add_argument("--show_modality", default="flair", choices=MODALITIES)
    ap.add_argument("--target_grade", type=int, default=None, help="0/1/2; default = predicted")
    ap.add_argument("--slices", type=int, nargs="*", default=None)
    ap.add_argument("--output_dir", default="outputs/xai_maps")
    a = ap.parse_args()

    ckpt = Path(a.checkpoint)
    cfg_path = Path(a.config) if a.config else ckpt.parent / "config_used.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACFXAIFLNet(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    pid = Path(a.patient).name
    rec = {"image": [str(Path(a.patient) / f"{pid}_{m}.nii.gz") for m in MODALITIES],
           "label": str(Path(a.patient) / f"{pid}_seg.nii.gz")}
    s = build_transforms(cfg["spatial_size"], train=False)(rec)
    mods = HETEROGENEITY_PRESETS[cfg.get("heterogeneity_level", "high")][a.hospital]
    x = (s["image"] * modality_mask(mods).view(-1, 1, 1, 1)).unsqueeze(0).to(device)
    lbl = s["label"].float().unsqueeze(0).to(device)

    with torch.enable_grad():
        out = model(x)
        tgt = None if a.target_grade is None else torch.tensor([a.target_grade], device=device)
        cam = model.gradcam(out["feat"], out["cls_logits"], target=tgt)
    cam = F.interpolate(cam.detach(), size=x.shape[2:], mode="trilinear", align_corners=False)
    prob = torch.softmax(out["cls_logits"].detach(), 1)[0].cpu().numpy()
    pred = int(prob.argmax())
    print(f"{pid}: predicted {GRADES[pred]} (p = {prob[pred]:.3f}); hospital {a.hospital} modalities {mods}")
    print("Grad-CAM IoU (thr 0.5):", {k: round(v, 4) for k, v in cam_iou(cam, lbl)[0].items()})

    vol = x[0, MODALITIES.index(a.show_modality)].cpu().numpy()
    if vol.std() == 0:                                # requested modality missing at this hospital
        vol = x[0, [MODALITIES.index(m) for m in mods][0]].cpu().numpy()
    camv = cam[0, 0].cpu().numpy()
    wt = lbl[0, 1].cpu().numpy()
    slices = a.slices or [int(np.argmax(wt.sum(axis=(0, 1))))]     # axial slice with largest WT area

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for z in slices:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(np.rot90(vol[:, :, z]), cmap="gray"); ax[0].set_title("MRI")
        ax[1].imshow(np.rot90(vol[:, :, z]), cmap="gray")
        ax[1].imshow(np.rot90(camv[:, :, z]), cmap="jet", alpha=0.45, vmin=0, vmax=1); ax[1].set_title("Grad-CAM")
        ax[2].imshow(np.rot90(vol[:, :, z]), cmap="gray")
        ax[2].contour(np.rot90(wt[:, :, z]), levels=[0.5], colors="lime"); ax[2].set_title("Ground-truth WT")
        for axi in ax:
            axi.axis("off")
        fig.suptitle(f"{pid} | hospital {a.hospital} | {GRADES[pred]} | slice {z}")
        fig.tight_layout()
        p = out_dir / f"{pid}_hosp{a.hospital}_slice{z}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print("saved", p)


if __name__ == "__main__":
    main()
