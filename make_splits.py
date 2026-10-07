"""
make_splits.py — create data/hospital_splits.json (patient-level assignments).

Procedure (Section 4, "Dataset Description" and "Stratified Splits"):
  1. 369 BraTS 2020 training subjects with WHO grade labels (II / III / IV, Table 2).
  2. Grade-stratified assignment to four virtual hospitals of size
       A = 90 (T1ce), B = 89 (T2), C = 95 (FLAIR), D = 95 (multi-modal).
  3. Inside each hospital, grade-stratified 70 : 10 : 20 train / val / test split.
All randomness uses a fixed seed (default 42), so the file is fully reproducible.

Input: a CSV with columns  subject_id,grade   (grade in {2,3,4}),
       e.g. data/grade_labels.csv (see data/README.md for how it was built).

    python scripts/make_splits.py --grades data/grade_labels.csv --out data/hospital_splits.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

import numpy as np

HOSPITAL_SIZES = {"A": 90, "B": 89, "C": 95, "D": 95}
HOSPITAL_MODALITY = {"A": "T1ce", "B": "T2", "C": "FLAIR", "D": "T1+T1ce+T2+FLAIR"}
RATIOS = (0.70, 0.10, 0.20)


def read_grades(path: str) -> dict[str, int]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    g = {r["subject_id"].strip(): int(r["grade"]) for r in rows}
    bad = {k: v for k, v in g.items() if v not in (2, 3, 4)}
    if bad:
        raise ValueError(f"grades must be 2, 3 or 4; offending rows: {list(bad.items())[:5]}")
    return g


def largest_remainder(total: int, fractions: list[float]) -> list[int]:
    raw = np.array(fractions) * total
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base))[: total - base.sum()]:
        base[i] += 1
    return base.tolist()


def stratified_partition(ids_by_grade: dict[int, list[str]], sizes: dict[str, int], rng) -> dict[str, list[str]]:
    """Split each grade stratum across bins proportionally to the bin sizes, then fix totals exactly."""
    total = sum(sizes.values())
    names = list(sizes)
    out = {n: [] for n in names}
    for grade, ids in sorted(ids_by_grade.items()):
        ids = list(ids)
        rng.shuffle(ids)
        counts = largest_remainder(len(ids), [sizes[n] / total for n in names])
        pos = 0
        for n, c in zip(names, counts):
            out[n] += ids[pos: pos + c]
            pos += c
    # move surplus subjects so each bin has exactly its target size
    for _ in range(total):
        over = [n for n in names if len(out[n]) > sizes[n]]
        under = [n for n in names if len(out[n]) < sizes[n]]
        if not over:
            break
        out[under[0]].append(out[over[0]].pop())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", required=True, help="CSV with subject_id,grade")
    ap.add_argument("--out", default="data/hospital_splits.json")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    grades = read_grades(a.grades)
    if len(grades) != sum(HOSPITAL_SIZES.values()):
        raise ValueError(f"expected 369 subjects, got {len(grades)}")
    rng = np.random.default_rng(a.seed)

    by_grade = defaultdict(list)
    for pid, g in sorted(grades.items()):
        by_grade[g].append(pid)
    hospitals = stratified_partition(by_grade, HOSPITAL_SIZES, rng)

    result = {"meta": {"dataset": "BraTS 2020 training set (369 subjects)",
                       "seed": a.seed, "ratios_train_val_test": RATIOS,
                       "stratified_by": "WHO grade (II/III/IV)",
                       "hospital_modality": HOSPITAL_MODALITY},
              "grades": grades, "hospitals": {}}
    for h, ids in hospitals.items():
        hb = defaultdict(list)
        for pid in sorted(ids):
            hb[grades[pid]].append(pid)
        n = len(ids)
        sz = dict(zip(("train", "val", "test"), largest_remainder(n, list(RATIOS))))
        parts = stratified_partition(hb, sz, rng)
        result["hospitals"][h] = {"modality": HOSPITAL_MODALITY[h], **{k: sorted(v) for k, v in parts.items()}}

    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)

    print(f"wrote {a.out}")
    for h, d in result["hospitals"].items():
        c = {p: dict(Counter(grades[i] for i in d[p])) for p in ("train", "val", "test")}
        print(f"  {h} ({d['modality']}): train {len(d['train'])}, val {len(d['val'])}, test {len(d['test'])}  grades {c}")


if __name__ == "__main__":
    main()
