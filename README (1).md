# ACF-XAI-FL — Code, Configuration and Patient-Level Splits

Reproducibility repository for

> **ACF-XAI-FL: Explainable and Privacy-Preserving Cross-Hospital Federated Learning for Brain Tumour MRI Analysis**
> *International Journal of Intelligent Engineering and Systems (IJIES)*

Repository: <https://github.com/jeevithsh-cmd/ACF_scripts>

---

## Contents

```
ACF_scripts/
├── configs/
│   └── default.yaml          # every hyper-parameter used in the paper (incl. seeds 42, 123, 456)
├── data/
│   ├── README.md             # how to obtain BraTS 2020 and build the files below
│   ├── grade_labels.csv      # subject_id -> WHO grade II/III/IV (Table 2)  [ADD: see grade_labels_TEMPLATE.csv]
│   └── hospital_splits.json  # patient-level split  [GENERATE with make_splits.py, then commit]
├── scripts/
│   ├── main_train.py         # federated training + evaluation (Algorithm 1)
│   ├── models.py             # 3-D ResNet-34 encoder, MMAM, U-Net decoder, heads
│   ├── fl_modules.py         # MDGP, TSMC, CEA, FTPMB
│   ├── noise_config.py       # round-level Gaussian perturbation (no privacy accounting)
│   ├── data.py               # BraTS loading, pre-processing, modality heterogeneity presets
│   ├── metrics.py            # DSC, HD95, accuracy / macro-F1 / AUC, Grad-CAM IoU
│   ├── make_splits.py        # builds data/hospital_splits.json (seeded, stratified)
│   ├── gradcam_xai.py        # Grad-CAM figures (Figure 4)
│   ├── summarize_results.py  # mean ± std tables over seeds, Wilcoxon tests
│   └── acf_xai_fl.m          # MATLAB figures from the saved outputs
├── run_all.sh                # runs every experiment in the paper
├── requirements.txt
└── LICENSE
```

## Paper ↔ code map

| Paper | Code |
|---|---|
| MMAM, Eqs. 10–18 | `models.MMAM` |
| MDGP, Eqs. 19–27 | `fl_modules.mdgp_alpha`, `fl_modules.mdgp_project`, `main_train.train_local` |
| TSMC, Eqs. 28–35 | `fl_modules.TSMCServer` |
| CEA, Eqs. 36–40 | `models.ACFXAIFLNet.gradcam / explanation_embedding`, `fl_modules.CEAState`, `fl_modules.cea_loss` |
| FTPMB, Eqs. 41–42 | `fl_modules.FTPMB` |
| Total loss, Eq. 43 | `main_train.train_local` |
| Round-level perturbation, Eq. 44 | `noise_config.perturb_update` |
| Algorithm 1 | `main_train.run` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10–3.13, PyTorch ≥ 2.0, MONAI ≥ 1.3. A GPU with ≥ 24 GB memory is
recommended for 160 × 160 × 128 volumes with batch size 4.

## Quick check (no data needed)

```bash
python scripts/main_train.py --dry-run
```

This runs 2 rounds on small synthetic volumes on the CPU and writes every output file to `outputs/dry_run/`.

## Full reproduction

```bash
# 1. data (see data/README.md)
python scripts/make_splits.py --grades data/grade_labels.csv --out data/hospital_splits.json

# 2. main result, 3 seeds
python scripts/main_train.py --config configs/default.yaml

# 3. everything (ablations, heterogeneity, sigma sensitivity, centralised bound) + tables
bash run_all.sh
```

Outputs per run: `outputs/<experiment>/seed_<s>/`
`config_used.yaml`, `history.csv` (validation curve), `test_per_patient.csv`,
`test_summary.json`, `final_model.pt`. Tables: `outputs/tables/*.csv`.

### Reproducing every table

| Paper item | Command / experiment name |
|---|---|
| Table 3, 4, 9, 10, Fig. 2–4 | `acf_xai_fl_full` |
| Table 5 (ablation) | `wo_mdgp`, `wo_tsmc`, `wo_cea`, `wo_ftpmb`, `wo_prox`, `wo_mmam`, `sigma_0`, `wo_mdgp_cea`, `fedavg` |
| Table 6 (rounds to 85 % of centralised) | `centralised`, `acf_xai_fl_full`, `fedavg` → `outputs/tables/convergence.csv` |
| Table 7 (heterogeneity) | `het_{low,medium,high,extreme}` and `het_*_fedavg` |
| Table 8 (σ sensitivity) | `sigma_0`, `acf_xai_fl_full` (σ = 0.01), `sigma_{0.05,0.10,0.20}` |
| Figure 4 | `python scripts/gradcam_xai.py --checkpoint outputs/acf_xai_fl_full/seed_42/final_model.pt --patient <subject folder> --hospital D` |

Any config value can be overridden: `--set key=value [key=value ...]`.

## Key settings (`configs/default.yaml`)

| Group | Values |
|---|---|
| Federation | K = 4, T = 100 rounds, E = 5 local epochs, B = 4 |
| Optimisation | Adam, lr = 1×10⁻⁴, weight decay = 1×10⁻⁴, gradient clip = 1.0, server lr η_s = 0.01 |
| Loss weights | λ1 = 0.5 (Dice), λ2 = 0.1 (CEA), λ3 = 0.05 (proto), λ4 = 0.01 (DC), λ5 = ρ = 0.01 (prox) |
| TSMC | γ = 0.9, μ = 0.8 |
| CEA | τ = 0.07, queue = 256, embedding dimension = 256 |
| Perturbation | σ = 0.01 (variance 1×10⁻⁴), C = 1.0, once per round |
| Seeds | 42, 123, 456 |

## Note on privacy

`noise_config.py` adds Gaussian noise **once per communication round** to the clipped client
model update. This is **not** DP-SGD, and the code performs **no** (ε, δ) privacy accounting.
In line with the paper, no formal differential-privacy guarantee is claimed. The
secure-aggregation layer described in the paper is conceptual and is not implemented here.

## Note on baselines

FedAvg and FedProx can be run from this code base (see `run_all.sh`). The SCAFFOLD, FedNova
and FedDyn numbers in the paper were produced with TensorFlow Federated v0.40.0, and FedXAI
with the authors' official repository (<https://github.com/zhaoyang-ai/FedXAI>), using the
hyper-parameters in Table 1 of the paper and the same `hospital_splits.json`.

## Citation

```bibtex
@article{acfxaifl2026,
  title   = {ACF-XAI-FL: Explainable and Privacy-Preserving Cross-Hospital Federated
             Learning for Brain Tumour MRI Analysis},
  author  = {Keerthi, D. S. and Jeevith, S. H. and Komala, K. and Ahamed, Ghouse Z. and Maanasa, M. R. and Shreedevi, P.},
  journal = {International Journal of Intelligent Engineering and Systems},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`. BraTS 2020 data are subject to the challenge's own terms of use.
