#!/usr/bin/env bash
# GPU 0 half of the 1.5B dev sweep: full-policy grid + steering alphas, at the
# plan's n_rollouts=4. The exit_only half runs concurrently on GPU 1
# (gpu1_1p5b_noop_and_baselines.sh); run analyze manually when both halves are
# done. Grid on dev ONLY. Completed points are skipped by policy hash.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="--config configs/base.yaml"

for TAU in 0.7 0.8 0.9 0.95; do for K in 1 2; do
  RC_SPLIT=dev python -m reasoncontrol.stages.run_controller $CFG \
    --set policy.kind=full --set policy.tau_exit=$TAU --set policy.patience_k=$K \
    --set datasets='[math_train,gsm8k]'
done; done
for A in 3 6 9; do
  RC_SPLIT=dev python -m reasoncontrol.stages.run_controller $CFG \
    --set policy.kind=steer_only --set policy.alpha=$A \
    --set datasets='[math_train,gsm8k]'
done
echo "GPU0 SWEEP HALF DONE (full + steer)"
