#!/usr/bin/env bash
# Prioritized 1.5B dev sweep: the plan's ~16-config grid (full/exit_only x
# tau x K) + noop + steer alphas, at n_rollouts=2 (walltime: measured ~2h/point
# at n=4 on the HF loop; deviation from "4 for sweeps" logged in STATUS.md).
# Grid on dev ONLY. Points already swept at n=4 are skipped by policy hash.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="--config configs/base.yaml"

for TAU in 0.7 0.8 0.9 0.95; do for K in 1 2; do
  RC_SPLIT=dev python -m reasoncontrol.stages.run_controller $CFG \
    --set policy.kind=full --set policy.tau_exit=$TAU --set policy.patience_k=$K \
    --set gen.n_rollouts=2 --set datasets='[math_train,gsm8k]'
  RC_SPLIT=dev python -m reasoncontrol.stages.run_controller $CFG \
    --set policy.kind=exit_only --set policy.tau_exit=$TAU --set policy.patience_k=$K \
    --set gen.n_rollouts=2 --set datasets='[math_train,gsm8k]'
done; done
RC_SPLIT=dev python -m reasoncontrol.stages.run_controller $CFG \
  --set policy.kind=noop --set gen.n_rollouts=2 --set datasets='[math_train,gsm8k]'
for A in 3 6 9; do
  RC_SPLIT=dev python -m reasoncontrol.stages.run_controller $CFG \
    --set policy.kind=steer_only --set policy.alpha=$A \
    --set gen.n_rollouts=2 --set datasets='[math_train,gsm8k]'
done
RC_SPLIT=dev python -m reasoncontrol.stages.analyze $CFG
echo "TRIM SWEEP DONE"
