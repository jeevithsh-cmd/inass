"""
summarize_results.py — aggregate the 3 seeds of every experiment into paper tables.

Reads outputs/<experiment>/seed_*/test_summary.json and history.csv and writes
  outputs/tables/segmentation.csv     (Table 3: WT/TC/ET DSC and HD95, mean +- std)
  outputs/tables/classification.csv   (Table 4: accuracy, macro-F1, AUC-ROC)
  outputs/tables/per_hospital.csv     (Table 9: per-hospital WT DSC)
  outputs/tables/xai_iou.csv          (Table 10: Grad-CAM IoU)
  outputs/tables/convergence.csv      (Table 6: rounds to 85 % of centralised WT DSC)
  outputs/tables/wilcoxon.csv         (paired Wilcoxon signed-rank on per-patient WT DSC
                                        vs. --reference experiment; seeds averaged per patient)

    python scripts/summarize_results.py --outputs outputs --reference acf_xai_fl_full --centralised centralised
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f}±{v.std(ddof=1) if len(v) > 1 else 0:.3f}"


def load(exp_dir: Path):
    seeds = sorted(exp_dir.glob("seed_*"))
    summ = [json.load(open(s / "test_summary.json")) for s in seeds if (s / "test_summary.json").exists()]
    return seeds, summ


def per_patient_wt(exp_dir: Path) -> dict[str, float]:
    acc = defaultdict(list)
    for s in exp_dir.glob("seed_*/test_per_patient.csv"):
        for r in csv.DictReader(open(s)):
            acc[r["pid"]].append(float(r["WT_DSC"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def curve(seed_dir: Path):
    r, wt = [], []
    for row in csv.DictReader(open(seed_dir / "history.csv")):
        if row["val_WT_DSC"]:
            r.append(int(row["round"]))
            wt.append(float(row["val_WT_DSC"]))
    return np.array(r), np.array(wt)


def write(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--reference", default="acf_xai_fl_full")
    ap.add_argument("--centralised", default="centralised")
    a = ap.parse_args()
    root = Path(a.outputs)
    exps = sorted(p for p in root.iterdir() if p.is_dir() and any(p.glob("seed_*/test_summary.json")))
    T = root / "tables"

    seg, cls, hosp, xai, conv = [], [], [], [], []
    cen_target = None
    if (root / a.centralised).exists():
        _, cs = load(root / a.centralised)
        cen_target = 0.85 * np.mean([s["segmentation_and_xai"]["WT_DSC"] for s in cs])
    for e in exps:
        seeds, S = load(e)
        g = lambda sec, k: [s[sec][k] for s in S]
        seg.append([e.name, len(S)] + [ms(g("segmentation_and_xai", f"{r}_{m}"))
                                        for m in ("DSC", "HD95") for r in ("WT", "TC", "ET")])
        cls.append([e.name, ms(g("classification", "Accuracy")), ms(g("classification", "Macro_F1")),
                    ms(g("classification", "AUC_ROC"))])
        hosp.append([e.name] + [ms([s["per_hospital"][h]["WT_DSC"] for s in S]) for h in "ABCD"
                                if h in S[0]["per_hospital"]])
        xai.append([e.name] + [ms(g("segmentation_and_xai", k)) for k in ("WT_IoU", "TC_IoU", "ET_IoU", "Mean_IoU")])
        if cen_target is not None:
            hits, finals = [], []
            for sd in seeds:
                r, wt = curve(sd)
                idx = np.where(wt >= cen_target)[0]
                hits.append(r[idx[0]] if len(idx) else np.nan)
                finals.append(wt[-1])
            conv.append([e.name, f"{cen_target:.3f}", ms(hits) if not np.isnan(hits).any() else "not reached",
                         ms(finals)])

    write(T / "segmentation.csv", ["experiment", "n_seeds", "WT_DSC", "TC_DSC", "ET_DSC",
                                    "WT_HD95", "TC_HD95", "ET_HD95"], seg)
    write(T / "classification.csv", ["experiment", "Accuracy_%", "Macro_F1", "AUC_ROC"], cls)
    write(T / "per_hospital.csv", ["experiment", "A_T1ce", "B_T2", "C_FLAIR", "D_multi"], hosp)
    write(T / "xai_iou.csv", ["experiment", "WT_IoU", "TC_IoU", "ET_IoU", "Mean_IoU"], xai)
    if conv:
        write(T / "convergence.csv", ["experiment", "85pct_centralised_WT", "rounds_to_target",
                                       "final_val_WT"], conv)

    ref = root / a.reference
    if ref.exists():
        rp = per_patient_wt(ref)
        rows = []
        for e in exps:
            if e.name == a.reference:
                continue
            op = per_patient_wt(e)
            common = sorted(set(rp) & set(op))
            if len(common) < 5:
                continue
            x, y = np.array([rp[k] for k in common]), np.array([op[k] for k in common])
            stat, p = wilcoxon(x, y)
            rows.append([a.reference, e.name, len(common), f"{x.mean():.4f}", f"{y.mean():.4f}", f"{stat:.1f}", f"{p:.3e}"])
        write(T / "wilcoxon.csv", ["reference", "compared", "n_patients", "ref_mean_WT", "cmp_mean_WT",
                                    "W", "p_value"], rows)


if __name__ == "__main__":
    main()
