#!/usr/bin/env bash
# End-to-end smoke on a tiny model. Needs huggingface.co reachable.
# Exercises every stage interface the real runs use (hf backend; judge stubbed).
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="--config configs/base.yaml configs/smoke.yaml"

python -m reasoncontrol.stages.prepare_data $CFG --datasets gsm8k
python -m reasoncontrol.stages.generate     $CFG --set gen.n_rollouts=2 --set datasets='[gsm8k]'
python -m reasoncontrol.stages.chunk        $CFG --set datasets='[gsm8k]'
python -m reasoncontrol.stages.capture      $CFG --set datasets='[gsm8k]'
python -m reasoncontrol.stages.forced_answer $CFG --set datasets='[gsm8k]' --audit
python -m reasoncontrol.stages.label_phase  $CFG --set datasets='[gsm8k]'
python -m reasoncontrol.stages.train_probes $CFG --set datasets='[gsm8k]' --datasets gsm8k
python -m reasoncontrol.stages.build_steering $CFG --datasets gsm8k
# two controller points + two baselines on dev, then analysis
RC_SPLIT=dev python -m reasoncontrol.stages.run_controller $CFG --set datasets='[gsm8k]' --set policy.kind=noop
RC_SPLIT=dev python -m reasoncontrol.stages.run_controller $CFG --set datasets='[gsm8k]' --set policy.kind=full --set policy.tau_exit=0.8
RC_SPLIT=dev python -m reasoncontrol.stages.run_baselines  $CFG --set datasets='[gsm8k]' --set policy.kind=static_budget --set policy.budget=64
RC_SPLIT=dev python -m reasoncontrol.stages.analyze        $CFG
echo "SMOKE OK"
