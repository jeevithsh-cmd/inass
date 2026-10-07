#!/usr/bin/env bash
# Reproduce every ACF-XAI-FL experiment in the paper (3 seeds each: 42, 123, 456).
# Each run writes outputs/<experiment_name>/seed_<s>/ ; then tables are built by
# scripts/summarize_results.py and figures by scripts/acf_xai_fl.m.
set -euo pipefail
cd "$(dirname "$0")"
CFG=configs/default.yaml
T="python scripts/main_train.py --config $CFG"

# ---------------------------------------------------------------- main result (Tables 3, 4, 9, 10; Figures 2-4)
$T --set experiment_name=acf_xai_fl_full

# ---------------------------------------------------------------- centralised upper bound (85 % target, Table 6)
$T --centralised --set experiment_name=centralised heterogeneity_level=low

# ---------------------------------------------------------------- FedAvg / FedProx reference within this code base
$T --set experiment_name=fedavg  use_mmam=false use_mdgp=false use_tsmc=false use_cea=false use_ftpmb=false use_prox=false use_noise=false
$T --set experiment_name=fedprox use_mmam=false use_mdgp=false use_tsmc=false use_cea=false use_ftpmb=false use_prox=true  use_noise=false lambda_prox=0.01

# ---------------------------------------------------------------- ablation (Table 5)
$T --set experiment_name=wo_mdgp     use_mdgp=false
$T --set experiment_name=wo_tsmc     use_tsmc=false
$T --set experiment_name=wo_cea      use_cea=false
$T --set experiment_name=wo_ftpmb    use_ftpmb=false
$T --set experiment_name=wo_prox     use_prox=false
$T --set experiment_name=wo_mmam     use_mmam=false
$T --set experiment_name=sigma_0     use_noise=false
$T --set experiment_name=wo_mdgp_cea use_mdgp=false use_cea=false

# ---------------------------------------------------------------- heterogeneity stress test (Table 7)
for L in low medium high extreme; do
  $T --set experiment_name=het_${L} heterogeneity_level=${L}
  $T --set experiment_name=het_${L}_fedavg heterogeneity_level=${L} \
         use_mmam=false use_mdgp=false use_tsmc=false use_cea=false use_ftpmb=false use_prox=false use_noise=false
done

# ---------------------------------------------------------------- perturbation sensitivity (Table 8)
for S in 0.05 0.10 0.20; do
  $T --set experiment_name=sigma_${S} noise_sigma=${S}
done

python scripts/summarize_results.py --outputs outputs --reference acf_xai_fl_full --centralised centralised
echo "Done. Tables in outputs/tables/. Run acf_xai_fl in MATLAB for figures."
