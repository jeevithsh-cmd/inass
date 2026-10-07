"""
data.py — BraTS 2020 federated data pipeline.

* Patient-level hospital + train/val/test assignments are read from
  data/hospital_splits.json (created by make_splits.py).
* Modality heterogeneity is simulated by giving each virtual hospital only its
  modality channels (missing channels are zero-filled).
* Pre-processing (Sec. 4): resize to 160x160x128, modality-wise z-score,
  random rotation +-10 deg, scaling +-10 %, translation +-5 mm, intensity x[0.9, 1.1].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ["t1", "t1ce", "t2", "flair"]          # channel order of the input tensor
GRADE_TO_IDX = {2: 0, 3: 1, 4: 2}                     # WHO Grade II / III / IV -> class index

# Per-hospital available modalities for each heterogeneity level (Table 7).
HETEROGENEITY_PRESETS = {
    "low":     {"A": MODALITIES, "B": MODALITIES, "C": MODALITIES, "D": MODALITIES},
    "medium":  {"A": ["t1ce"], "B": ["t2"], "C": MODALITIES, "D": MODALITIES},
    "high":    {"A": ["t1ce"], "B": ["t2"], "C": ["flair"], "D": MODALITIES},   # main setting
    "extreme": {"A": ["t1ce"], "B": ["t2"], "C": ["flair"], "D": ["t1"]},
}
HOSPITALS = ["A", "B", "C", "D"]


def load_splits(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def modality_mask(mods: list[str]) -> torch.Tensor:
    return torch.tensor([1.0 if m in mods else 0.0 for m in MODALITIES])


# --------------------------------------------------------------------------- real BraTS
def _record(root: Path, pid: str, grade: int) -> dict:
    d = root / pid
    return {"image": [str(d / f"{pid}_{m}.nii.gz") for m in MODALITIES],
            "label": str(d / f"{pid}_seg.nii.gz"),
            "grade": GRADE_TO_IDX[int(grade)], "pid": pid}


def build_transforms(spatial_size, train: bool):
    from monai import transforms as T
    xs = [
        T.LoadImaged(keys=["image", "label"]),
        T.EnsureChannelFirstd(keys=["image", "label"]),
        T.Orientationd(keys=["image", "label"], axcodes="RAS"),
        T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),   # -> TC, WT, ET
        T.Resized(keys=["image"], spatial_size=spatial_size, mode="trilinear"),
        T.Resized(keys=["label"], spatial_size=spatial_size, mode="nearest"),
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if train:
        xs += [
            T.RandAffined(keys=["image", "label"], prob=0.5,
                          rotate_range=(np.deg2rad(10),) * 3,
                          scale_range=(0.1,) * 3,
                          translate_range=(5,) * 3,             # 1 mm voxels -> +-5 mm
                          mode=("bilinear", "nearest"), padding_mode="zeros"),
            T.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),   # x[0.9, 1.1]
        ]
    xs += [T.EnsureTyped(keys=["image", "label"], track_meta=False)]
    return T.Compose(xs)


class MaskedDataset(Dataset):
    """Wraps a dataset and zero-fills the modality channels a hospital does not have."""

    def __init__(self, base, mask: torch.Tensor, client_id: int):
        self.base, self.mask, self.cid = base, mask, client_id

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = dict(self.base[i])
        s["image"] = s["image"] * self.mask.view(-1, 1, 1, 1)
        s["label"] = s["label"].float()
        s["client"] = self.cid
        return s


def build_federated_datasets(cfg: dict):
    """Returns dict: hospital -> {'train','val','test': Dataset}, plus modality table."""
    from monai.data import CacheDataset
    splits = load_splits(cfg["split_file"])
    preset = HETEROGENEITY_PRESETS[cfg.get("heterogeneity_level", "high")]
    root = Path(cfg["data_root"])
    grades = splits["grades"]
    out = {}
    for cid, h in enumerate(HOSPITALS[: cfg["num_clients"]]):
        mask = modality_mask(preset[h])
        out[h] = {}
        for part in ("train", "val", "test"):
            recs = [_record(root, pid, grades[pid]) for pid in splits["hospitals"][h][part]]
            base = CacheDataset(recs, build_transforms(cfg["spatial_size"], part == "train"),
                                cache_rate=cfg.get("cache_rate", 0.0), num_workers=cfg.get("num_workers", 0))
            out[h][part] = MaskedDataset(base, mask, cid)
    return out, preset


# --------------------------------------------------------------------------- synthetic (dry run)
class SyntheticBraTS(Dataset):
    """Random 4-channel volumes with a spherical 'tumour'; used only for --dry-run smoke tests."""

    def __init__(self, n: int, size=(48, 48, 48), seed: int = 0):
        g = np.random.default_rng(seed)
        self.items = []
        for i in range(n):
            img = g.normal(0, 1, (4, *size)).astype(np.float32)
            lbl = np.zeros((3, *size), np.float32)
            c = g.integers(14, size[0] - 14, 3)
            r = g.integers(5, 10)
            zz, yy, xx = np.ogrid[: size[0], : size[1], : size[2]]
            ball = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= r ** 2
            core = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= (r // 2 + 1) ** 2
            grade = int(g.integers(0, 3))
            img[:, ball] += 2.0 + grade
            lbl[1, ball] = 1          # WT
            lbl[0, core] = 1          # TC
            lbl[2, core] = 1          # ET
            self.items.append({"image": torch.from_numpy(img), "label": torch.from_numpy(lbl),
                               "grade": grade, "pid": f"SYN_{seed}_{i:03d}"})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def build_synthetic_datasets(cfg: dict):
    preset = HETEROGENEITY_PRESETS[cfg.get("heterogeneity_level", "high")]
    out = {}
    for cid, h in enumerate(HOSPITALS[: cfg["num_clients"]]):
        mask = modality_mask(preset[h])
        out[h] = {p: MaskedDataset(SyntheticBraTS(n, seed=100 * cid + k), mask, cid)
                  for k, (p, n) in enumerate([("train", 4), ("val", 2), ("test", 2)])}
    return out, preset
