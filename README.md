# Data

## 1. BraTS 2020

Download the BraTS 2020 training data (369 subjects) from the official challenge page
<https://www.med.upenn.edu/cbica/brats2020/data.html> and unpack it so that

```
data/BraTS2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_{t1,t1ce,t2,flair,seg}.nii.gz
...
```

The imaging data cannot be redistributed and is therefore not part of this repository.

## 2. `grade_labels.csv` — tumour-grade labels (Table 2)

Columns: `subject_id,grade` with `grade ∈ {2, 3, 4}` (WHO Grade II / III / IV).
LGG subjects in the BraTS `name_mapping.csv` are Grade II; HGG subjects were sub-divided into
Grade III and Grade IV from the TCIA histopathology records (Section 4, "Tumour Grade
Classification Labels"). Distribution: 76 / 82 / 211 subjects.

## 3. `hospital_splits.json` — patient-level split assignments

Generated deterministically by

```bash
python scripts/make_splits.py --grades data/grade_labels.csv --out data/hospital_splits.json --seed 42
```

Structure:

```json
{
  "meta":   { "seed": 42, "ratios_train_val_test": [0.7, 0.1, 0.2], ... },
  "grades": { "BraTS20_Training_001": 4, ... },
  "hospitals": {
    "A": { "modality": "T1ce",  "train": [...63 ids], "val": [...9], "test": [...18] },
    "B": { "modality": "T2",    "train": [...62 ids], "val": [...9], "test": [...18] },
    "C": { "modality": "FLAIR", "train": [...67 ids], "val": [...9], "test": [...19] },
    "D": { "modality": "T1+T1ce+T2+FLAIR", "train": [...67 ids], "val": [...9], "test": [...19] }
  }
}
```

Hospital sizes 90 / 89 / 95 / 95; training-set sizes 63 / 62 / 67 / 67 give the
mini-batch sampling ratios q = 4/63, 4/62, 4/67, 4/67 quoted in the paper.
