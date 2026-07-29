#!/usr/bin/env bash
# GPU 1 work that needs NO sweep-selected operating point:
#  1) 1.5B vLLM dev baselines (budget grids etc., n=2 to pair with trim sweep)
#  2) 1.5B HF-loop noop TEST runs, 8 seeds (headline reference condition)
# Test runs here are protocol-safe: noop has no tunables, so nothing is
# selected on test.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="--config configs/base.yaml"

for B in 1024 2048 4096 8192; do
  RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG \
    --set policy.kind=static_budget --set policy.budget=$B \
    --set gen.n_rollouts=2 --set datasets='[math_train,gsm8k]'
  RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG \
    --set policy.kind=budget_prompt --set policy.budget=$B \
    --set gen.n_rollouts=2 --set datasets='[math_train,gsm8k]'
done
RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG \
  --set policy.kind=noop --set gen.n_rollouts=2 --set datasets='[math_train,gsm8k]'
RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG \
  --set policy.kind=concise_prompt --set gen.n_rollouts=2 --set datasets='[math_train,gsm8k]'
RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG \
  --set policy.kind=trial_decode --set gen.n_rollouts=2 --set datasets='[math_train,gsm8k]'

for S in 0 1 2 3 4 5 6 7; do
  RC_SPLIT=test python -m reasoncontrol.stages.run_controller $CFG \
    --set policy.kind=noop --set seed=$S --set gen.n_rollouts=1 \
    --set datasets='[math500,gsm8k,aime,gpqa_diamond]'
done
echo "GPU1 NOOP+BASELINES DONE"
