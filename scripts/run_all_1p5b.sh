#!/usr/bin/env bash
# Full 1.5B pipeline (GPU box). Stages are idempotent/resumable; rerun freely.
# GPU pinning: run this under CUDA_VISIBLE_DEVICES=0 while the 7B pipeline
# runs under CUDA_VISIBLE_DEVICES=1.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="--config configs/base.yaml"

python -m reasoncontrol.stages.prepare_data $CFG
python -m reasoncontrol.stages.generate     $CFG
python -m reasoncontrol.stages.chunk        $CFG
python -m reasoncontrol.stages.capture      $CFG
python -m reasoncontrol.stages.forced_answer $CFG --audit
python -m reasoncontrol.stages.label_phase  $CFG
python -m reasoncontrol.stages.train_probes $CFG
python -m reasoncontrol.stages.build_steering $CFG --datasets math_train

# ---- dev sweeps (GRID ON DEV ONLY; see plan) --------------------------------
for TAU in 0.7 0.8 0.9 0.95; do for K in 1 2 3; do
  RC_SPLIT=dev python -m reasoncontrol.stages.run_controller $CFG \
    --set policy.kind=full --set policy.tau_exit=$TAU --set policy.patience_k=$K \
    --set datasets='[math_train,gsm8k]'
  RC_SPLIT=dev python -m reasoncontrol.stages.run_controller $CFG \
    --set policy.kind=exit_only --set policy.tau_exit=$TAU --set policy.patience_k=$K \
    --set datasets='[math_train,gsm8k]'
done; done
RC_SPLIT=dev python -m reasoncontrol.stages.run_controller $CFG --set policy.kind=noop --set datasets='[math_train,gsm8k]'
for A in 3 6 9; do
  RC_SPLIT=dev python -m reasoncontrol.stages.run_controller $CFG \
    --set policy.kind=steer_only --set policy.alpha=$A --set datasets='[math_train,gsm8k]'
done
for B in 1024 2048 4096 8192; do
  RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG \
    --set policy.kind=static_budget --set policy.budget=$B --set datasets='[math_train,gsm8k]'
  RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG \
    --set policy.kind=budget_prompt --set policy.budget=$B --set datasets='[math_train,gsm8k]'
done
RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG --set policy.kind=noop --set datasets='[math_train,gsm8k]'
RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG --set policy.kind=concise_prompt --set datasets='[math_train,gsm8k]'
RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines $CFG --set policy.kind=trial_decode --set datasets='[math_train,gsm8k]'
RC_SPLIT=dev python -m reasoncontrol.stages.analyze $CFG

echo "Dev sweep done. Select k operating points per family (1-SE rule) from"
echo "runs/analysis, then run test with 8 seeds for headliners, e.g.:"
echo "  for S in 0 1 2 3 4 5 6 7; do RC_SPLIT=test python -m reasoncontrol.stages.run_controller \\"
echo "    $CFG --set seed=\$S --set policy.kind=full --set policy.tau_exit=<T*> \\"
echo "    --set policy.patience_k=<K*> --set datasets='[math500,gsm8k,aime,gpqa_diamond]'; done"
