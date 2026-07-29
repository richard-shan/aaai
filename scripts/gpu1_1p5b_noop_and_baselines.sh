#!/usr/bin/env bash
# GPU 1 (freed after the 7B's banked first point; 7B resumes later as a
# transfer study at 1.5B-selected operating points):
#  1) 1.5B vLLM dev baselines (fast)
#  2) exit_only half of the 1.5B dev sweep (n=4, pairs with GPU 0's full half)
#  3) controller-loop noop dev reference (n=4)
#  4) 1.5B HF-loop noop TEST, 8 seeds (protocol-safe now: noop has no tunables)
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="--config configs/base.yaml"

for B in 1024 2048 4096 8192; do
  RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG \
    --set policy.kind=static_budget --set policy.budget=$B --set datasets='[math_train,gsm8k]'
  RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG \
    --set policy.kind=budget_prompt --set policy.budget=$B --set datasets='[math_train,gsm8k]'
done
RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG \
  --set policy.kind=noop --set datasets='[math_train,gsm8k]'
RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG \
  --set policy.kind=concise_prompt --set datasets='[math_train,gsm8k]'
RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG \
  --set policy.kind=trial_decode --set datasets='[math_train,gsm8k]'

for TAU in 0.7 0.8 0.9 0.95; do for K in 1 2; do
  RC_SPLIT=dev python -m reasoncontrol.stages.run_controller $CFG \
    --set policy.kind=exit_only --set policy.tau_exit=$TAU --set policy.patience_k=$K \
    --set datasets='[math_train,gsm8k]'
done; done
RC_SPLIT=dev python -m reasoncontrol.stages.run_controller $CFG \
  --set policy.kind=noop --set datasets='[math_train,gsm8k]'

for S in 0 1 2 3 4 5 6 7; do
  RC_SPLIT=test python -m reasoncontrol.stages.run_controller $CFG \
    --set policy.kind=noop --set seed=$S --set gen.n_rollouts=1 \
    --set datasets='[math500,gsm8k,aime,gpqa_diamond]'
done
echo "GPU1 QUEUE DONE (baselines + exit_only half + noop dev/test)"
