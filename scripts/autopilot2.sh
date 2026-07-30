#!/usr/bin/env bash
# ReasonControl AUTOPILOT v2 — supersedes scripts/autopilot.sh after the
# 2026-07-30 grading defect (docs/RESULTS.md "MEASUREMENT DEFECT").
#
# What changed vs v1, and why:
#  1. gen.max_answer_tokens=2048 on every remaining run. v1 ran the HF
#     controller under a 512-token answer cap (loop.py:430) while the vLLM
#     baselines got a joint max_think+max_answer budget (baselines.py:45), so
#     verbose conditions lost their answers to truncation and the controller
#     looked worse than it is. 2048 is above the observed p99 answer length
#     (~900 tokens), so NEITHER engine truncates and the asymmetry is moot
#     without touching baseline code.
#  2. Test runs use the operating point selected from RE-GRADED dev data
#     (scripts/select_from_regrade.py): exit_only tau=0.7/K=2, not the
#     tau=0.7/K=1 point v1 was testing under the broken extractor.
#  3. steer_only alphas 6 and 9 are dropped (futility + billing): with correct
#     grading the full policy is 0.59-0.64 vs exit_only 0.78-0.82 on MATH dev,
#     so steering is already clearly dominated. alpha=3 (finishing under v1)
#     carries the pre-registered paired acceptance test.
#  4. Test seeds are split across both GPUs (v1 put all 8 on GPU1).
#  5. Final analyze is preceded by a full offline re-grade of both models.
#
# HASH TRAPS (policy_hash covers PolicyCfg only, NOT gen.*, so a 2048-budget
# rerun would silently skip a 512-budget file of the same policy):
#   - HF noop  : --set policy.min_chunks=5 (v1, 512) / =6 (v2, 2048)  [inert]
#   - exit_only: --set policy.alpha=6.5 for the 2048 DEV control       [inert,
#     ExitOnlyPolicy has use_steer=False]. TEST runs need no disambiguator —
#     no test file exists for hash 079322feb6.
#
# Idempotent; after any crash relaunch detached with no args:
#   cd /lambda/nfs/aaai/aaai && nohup setsid bash scripts/autopilot2.sh \
#     >> runs/logs/autopilot.log 2>&1 < /dev/null & disown
# Args: $1 = PID of an in-flight GPU0 stage python to wait for (optional).
set -u
cd /lambda/nfs/aaai/aaai
export PATH=/home/ubuntu/venvs/rc/bin:$PATH
export HF_HOME=/home/ubuntu/hf HF_TOKEN=$(cat /home/ubuntu/.hf_token)
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG="configs/base.yaml"
CFG7="configs/base.yaml configs/models/r1_qwen_7b.yaml"
SEL=runs/r1_qwen_1p5b/analysis/selection_dev.json
ANS=2048
WAIT0=${1:-0}
mkdir -p runs/logs

log(){ echo "[autopilot2 $(date -u +%FT%TZ)] $*"; }
snap(){ bash scripts/autopilot_snapshot.sh "$1" || true; }
step(){ local name="$1"; shift
  log "START $name"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then log "RETRY $name (rc=$rc)"; sleep 180; "$@"; rc=$?; fi
  if [ $rc -ne 0 ]; then log "STEPFAIL $name rc=$rc"
    echo "$(date -u +%FT%TZ) $name rc=$rc" >> runs/AUTOPILOT_FAILURES
  else log "OK $name"; fi
  return 0; }
waitpid(){ [ "${1:-0}" -gt 1 ] 2>/dev/null || return 0
  log "waiting for in-flight PID $1"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done; }
waitfile(){ log "waiting for $1"; while [ ! -e "$1" ]; do sleep 120; done; }
selget(){ python -c "import json;d=json.load(open('$SEL'));print(d['families']['$1']['tuned_params']['$2'])" 2>/dev/null; }

# TEST helpers. AIME runs separately at the 24k think cap with a reduced batch
# (KV-cache memory). Every run carries the 2048 answer budget.
ctl_test(){ local GPU=$1 MCFG=$2 S=$3 AB=$4 MB=$5; shift 5
  step "ctl-test[g$GPU s$S $*]" env RC_SPLIT=test CUDA_VISIBLE_DEVICES=$GPU \
    python -m reasoncontrol.stages.run_controller --config $MCFG \
    --set seed=$S --set gen.n_rollouts=1 --set gen.max_answer_tokens=$ANS \
    --set gen.batch_size=$MB \
    --set datasets='[math500,gsm8k,gpqa_diamond]' "$@"
  step "ctl-test-aime[g$GPU s$S $*]" env RC_SPLIT=test CUDA_VISIBLE_DEVICES=$GPU \
    python -m reasoncontrol.stages.run_controller --config $MCFG \
    --set seed=$S --set gen.n_rollouts=1 --set gen.max_think_tokens=24576 \
    --set gen.max_answer_tokens=$ANS --set gen.batch_size=$AB \
    --set datasets='[aime]' "$@"
}
base_test(){ local GPU=$1 MCFG=$2 S=$3; shift 3
  step "base-test[g$GPU s$S $*]" env RC_SPLIT=test CUDA_VISIBLE_DEVICES=$GPU \
    python -m reasoncontrol.stages.run_baselines --config $MCFG \
    --set seed=$S --set gen.n_rollouts=1 --set gen.max_answer_tokens=$ANS \
    --set datasets='[math500,gsm8k,gpqa_diamond]' "$@"
  step "base-test-aime[g$GPU s$S $*]" env RC_SPLIT=test CUDA_VISIBLE_DEVICES=$GPU \
    python -m reasoncontrol.stages.run_baselines --config $MCFG \
    --set seed=$S --set gen.n_rollouts=1 --set gen.max_think_tokens=24576 \
    --set gen.max_answer_tokens=$ANS --set datasets='[aime]' "$@"
}

T=$(selget exit_only tau_exit); K=$(selget exit_only patience_k)
BS=$(selget static_budget budget); BP=$(selget budget_prompt budget)

gpu0_chain(){
  waitpid "$WAIT0"          # steer_only alpha=3 dev, inherited from v1
  snap "v2 start: steer alpha=3 dev complete"

  # Answer-budget control: the SAME two conditions at 2048 that were run at
  # 512, so the cap's effect is measured rather than assumed. Inert overrides
  # keep the policy hashes distinct from the 512 runs.
  step "noop-dev-2048" env RC_SPLIT=dev CUDA_VISIBLE_DEVICES=0 \
    python -m reasoncontrol.stages.run_controller --config $CFG \
    --set policy.kind=noop --set policy.min_chunks=6 \
    --set gen.max_answer_tokens=$ANS --set datasets='[math_train,gsm8k]'
  step "exit-dev-2048" env RC_SPLIT=dev CUDA_VISIBLE_DEVICES=0 \
    python -m reasoncontrol.stages.run_controller --config $CFG \
    --set policy.kind=exit_only --set policy.tau_exit=$T \
    --set policy.patience_k=$K --set policy.alpha=6.5 \
    --set gen.max_answer_tokens=$ANS --set datasets='[math_train,gsm8k]'
  snap "answer-budget control at 2048 (noop + exit_only @ selected point)"

  # Pre-registered paired steering acceptance + D6 (512-budget family; the
  # steer_only sweep and its noop reference share that budget).
  step "steering-acceptance" python scripts/steering_acceptance.py
  snap "steering acceptance + D6 decision"

  for S in 4 5 6 7; do
    ctl_test 0 "$CFG" $S 32 48 --set policy.kind=exit_only \
      --set policy.tau_exit=$T --set policy.patience_k=$K
  done
  snap "1.5B exit_only test seeds 4-7 @ tau=$T k=$K (answer budget $ANS)"

  # Same-engine noop TEST reference (3 seeds; headline noop = 8-seed vLLM).
  for S in 0 1 2; do
    ctl_test 0 "$CFG" $S 32 48 --set policy.kind=noop --set policy.min_chunks=6
  done
  snap "1.5B HF-noop test reference (3 seeds, budget $ANS)"

  step "interp-1p5b" env CUDA_VISIBLE_DEVICES=0 \
    python -m reasoncontrol.stages.interp --config $CFG
  touch runs/.ap2_gpu0_done
  log "GPU0 CHAIN DONE"
}

gpu1_chain(){
  for S in 0 1 2 3; do
    ctl_test 1 "$CFG" $S 32 48 --set policy.kind=exit_only \
      --set policy.tau_exit=$T --set policy.patience_k=$K
  done
  snap "1.5B exit_only test seeds 0-3 @ tau=$T k=$K (answer budget $ANS)"

  for S in 0 1 2 3 4 5 6 7; do
    [ -n "$BS" ] && base_test 1 "$CFG" $S \
      --set policy.kind=static_budget --set policy.budget=$BS
    [ -n "$BP" ] && base_test 1 "$CFG" $S \
      --set policy.kind=budget_prompt --set policy.budget=$BP
    base_test 1 "$CFG" $S --set policy.kind=noop
    base_test 1 "$CFG" $S --set policy.kind=concise_prompt
    base_test 1 "$CFG" $S --set policy.kind=trial_decode
  done
  snap "1.5B vLLM baseline test suite (8 seeds, budget $ANS)"

  # 7B transfer at the 1.5B-selected point (4 seeds) + vLLM refs + 1 HF-noop.
  for S in 0 1 2 3; do
    ctl_test 1 "$CFG7" $S 16 16 --set policy.kind=exit_only \
      --set policy.tau_exit=$T --set policy.patience_k=$K
    base_test 1 "$CFG7" $S --set policy.kind=noop
    [ -n "$BS" ] && base_test 1 "$CFG7" $S \
      --set policy.kind=static_budget --set policy.budget=$BS
  done
  ctl_test 1 "$CFG7" 0 16 16 --set policy.kind=noop --set policy.min_chunks=6
  snap "7B transfer + references (budget $ANS)"

  step "interp-7b" env CUDA_VISIBLE_DEVICES=1 \
    python -m reasoncontrol.stages.interp --config $CFG7
  touch runs/.ap2_gpu1_done
  log "GPU1 CHAIN DONE"
}

log "AUTOPILOT2 START (wait0=$WAIT0) point: exit_only tau=$T k=$K, B*=$BS/$BP, answers=$ANS"
if [ -z "$T" ] || [ -z "$K" ]; then
  log "FATAL: no exit_only selection in $SEL"
  echo "$(date -u +%FT%TZ) autopilot2 no-selection" >> runs/AUTOPILOT_FAILURES
  exit 1
fi
gpu0_chain >> runs/logs/autopilot_gpu0.log 2>&1 &
P0=$!
gpu1_chain >> runs/logs/autopilot_gpu1.log 2>&1 &
P1=$!
log "chains launched: gpu0=$P0 gpu1=$P1"
wait $P0 $P1

# Re-grade everything offline before analysis so every reported number uses the
# fixed extractor (and so the 512-vs-2048 comparison is available).
step "regrade-1p5b" python scripts/regrade.py --model-root runs/r1_qwen_1p5b
step "regrade-7b" python scripts/regrade.py --model-root runs/r1_qwen_7b \
  --tokenizer deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
step "analyze-dev" env RC_SPLIT=dev python -m reasoncontrol.stages.analyze --config $CFG
step "analyze-test" env RC_SPLIT=test python -m reasoncontrol.stages.analyze --config $CFG
step "analyze-7b-test" env RC_SPLIT=test python -m reasoncontrol.stages.analyze --config $CFG7
NF=$(cat runs/AUTOPILOT_FAILURES 2>/dev/null | wc -l)
{ echo "# ReasonControl: ALL EXPERIMENTS DONE"
  echo
  echo "- Finished (UTC): $(date -u +%FT%TZ)"
  echo "- Step failures logged: $NF (see docs/AUTOLOG.md)"
  echo "- Test point: exit_only tau=$T k=$K; answer budget $ANS tokens."
  echo "- Numbers regraded with the fixed extractor (docs/RESULTS.md)."
  echo "- Safe to terminate the GPU instance AFTER verifying this commit is on GitHub main."
} > EXPERIMENTS_DONE.md
date -u > runs/AUTOPILOT_DONE
snap "AUTOPILOT2 COMPLETE (failures=$NF) — safe to terminate GPU instance"
log "AUTOPILOT COMPLETE failures=$NF"
