#!/usr/bin/env bash
# ReasonControl AUTOPILOT — owns BOTH GPU queues end-to-end:
#   dev sweeps (remaining) -> 1-SE selection + closed-loop audit -> test runs
#   -> 7B transfer -> steering acceptance + D6 -> interp -> final analyze.
# Commits+pushes artifacts at every milestone (autopilot_snapshot.sh) and ends
# by committing EXPERIMENTS_DONE.md at the repo root.
#
# Every step is idempotent (per-dataset results files under policy-hash dirs),
# so after ANY crash/reboot just relaunch detached with no args:
#   cd /lambda/nfs/aaai/aaai && nohup setsid bash scripts/autopilot.sh \
#     >> runs/logs/autopilot.log 2>&1 < /dev/null & disown
# Args: $1/$2 = PIDs of in-flight GPU0/GPU1 stage pythons to wait for (optional).
#
# NOTE: HF-loop noop runs use --set policy.min_chunks=5 (behaviorally inert for
# noop) ONLY to keep their policy hash distinct from the vLLM baseline noop —
# otherwise the two engines silently skip each other's output files.
set -u
cd /lambda/nfs/aaai/aaai
export PATH=/home/ubuntu/venvs/rc/bin:$PATH
export HF_HOME=/home/ubuntu/hf HF_TOKEN=$(cat /home/ubuntu/.hf_token)
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG="configs/base.yaml"
CFG7="configs/base.yaml configs/models/r1_qwen_7b.yaml"
SEL=runs/r1_qwen_1p5b/analysis/selection_dev.json
WAIT0=${1:-0}; WAIT1=${2:-0}
mkdir -p runs/logs

log(){ echo "[autopilot $(date -u +%FT%TZ)] $*"; }
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
selget(){ python -c "import json;d=json.load(open('$SEL'));print(d['families']['$1']['params']['$2'])" 2>/dev/null; }

# TEST-split helpers. AIME runs in its own invocation at the 24k think cap
# with a reduced batch (KV-cache memory); other test sets use defaults.
ctl_test(){ local GPU=$1 MCFG=$2 S=$3 AB=$4; shift 4
  step "ctl-test[g$GPU s$S $*]" env RC_SPLIT=test CUDA_VISIBLE_DEVICES=$GPU \
    python -m reasoncontrol.stages.run_controller --config $MCFG \
    --set seed=$S --set gen.n_rollouts=1 \
    --set datasets='[math500,gsm8k,gpqa_diamond]' "$@"
  step "ctl-test-aime[g$GPU s$S $*]" env RC_SPLIT=test CUDA_VISIBLE_DEVICES=$GPU \
    python -m reasoncontrol.stages.run_controller --config $MCFG \
    --set seed=$S --set gen.n_rollouts=1 --set gen.max_think_tokens=24576 \
    --set gen.batch_size=$AB --set datasets='[aime]' "$@"
}
base_test(){ local GPU=$1 MCFG=$2 S=$3; shift 3
  step "base-test[g$GPU s$S $*]" env RC_SPLIT=test CUDA_VISIBLE_DEVICES=$GPU \
    python -m reasoncontrol.stages.run_baselines --config $MCFG \
    --set seed=$S --set gen.n_rollouts=1 \
    --set datasets='[math500,gsm8k,gpqa_diamond]' "$@"
  step "base-test-aime[g$GPU s$S $*]" env RC_SPLIT=test CUDA_VISIBLE_DEVICES=$GPU \
    python -m reasoncontrol.stages.run_baselines --config $MCFG \
    --set seed=$S --set gen.n_rollouts=1 --set gen.max_think_tokens=24576 \
    --set datasets='[aime]' "$@"
}

gpu0_chain(){
  waitpid "$WAIT0"
  # Finish the full-policy grid, futility-trimmed to 5 points (0.7/0.8 x K1/K2
  # + 0.9/K1): banked points skip instantly by policy hash. Documented
  # deviation: dev curve is flat 0.476-0.482 and dominated; 0.9/K2 + 0.95s cut.
  for TAU in 0.7 0.8; do for K in 1 2; do
    step "full-dev[t$TAU k$K]" env RC_SPLIT=dev CUDA_VISIBLE_DEVICES=0 \
      python -m reasoncontrol.stages.run_controller --config $CFG \
      --set policy.kind=full --set policy.tau_exit=$TAU \
      --set policy.patience_k=$K --set datasets='[math_train,gsm8k]'
  done; done
  step "full-dev[t0.9 k1]" env RC_SPLIT=dev CUDA_VISIBLE_DEVICES=0 \
    python -m reasoncontrol.stages.run_controller --config $CFG \
    --set policy.kind=full --set policy.tau_exit=0.9 --set policy.patience_k=1 \
    --set datasets='[math_train,gsm8k]'
  snap "GPU0 full-policy dev grid done (futility-trimmed to 5 pts)"
  for A in 3 6 9; do
    step "steer-dev[a$A]" env RC_SPLIT=dev CUDA_VISIBLE_DEVICES=0 \
      python -m reasoncontrol.stages.run_controller --config $CFG \
      --set policy.kind=steer_only --set policy.alpha=$A \
      --set datasets='[math_train,gsm8k]'
  done
  step "select-full-steer" python scripts/select_operating_points.py \
    --families full,steer_only
  snap "GPU0 steer_only alphas done"
  touch runs/.ap_gpu0_dev_done

  waitfile runs/.ap_gpu1_dev_done   # HF-noop dev reference comes from GPU1
  step "steering-acceptance" python scripts/steering_acceptance.py
  snap "steering acceptance + D6 decision"

  # HF-loop noop TEST reference (engine control), trimmed 8->3 seeds
  # (documented deviation; headline noop comparison = 8-seed vLLM runs).
  for S in 0 1 2; do
    ctl_test 0 "$CFG" $S 32 --set policy.kind=noop --set policy.min_chunks=5
  done
  snap "1.5B HF-noop test reference (3 seeds)"

  # Full policy ships to test ONLY if steering was accepted (pre-registered).
  local ACCEPTED FT FK
  ACCEPTED=$(python -c "import json;print(int(json.load(open('runs/r1_qwen_1p5b/analysis/steering_acceptance.json')).get('any_accepted', False)))" 2>/dev/null || echo 0)
  if [ "$ACCEPTED" = "1" ]; then
    FT=$(selget full tau_exit); FK=$(selget full patience_k)
    if [ -n "$FT" ] && [ -n "$FK" ]; then
      log "steering ACCEPTED -> full-policy test at tau=$FT k=$FK (8 seeds)"
      for S in 0 1 2 3 4 5 6 7; do
        ctl_test 0 "$CFG" $S 32 --set policy.kind=full \
          --set policy.tau_exit=$FT --set policy.patience_k=$FK
      done
      snap "1.5B full-policy test (8 seeds)"
    fi
  else
    log "steering REJECTED -> D6 exit-led headline; full policy not run on test"
  fi

  # 7B transfer at the 1.5B-selected exit_only point (4 seeds; documented).
  waitfile "$SEL"
  local T7 K7
  T7=$(selget exit_only tau_exit); K7=$(selget exit_only patience_k)
  if [ -n "$T7" ] && [ -n "$K7" ]; then
    for S in 0 1 2 3; do
      ctl_test 0 "$CFG7" $S 16 --set policy.kind=exit_only \
        --set policy.tau_exit=$T7 --set policy.patience_k=$K7
    done
    snap "7B transfer: exit_only test @ tau=$T7 k=$K7 (4 seeds)"
  else
    log "STEPFAIL 7b-transfer: no exit_only selection"
    echo "$(date -u +%FT%TZ) 7b-transfer no-selection" >> runs/AUTOPILOT_FAILURES
  fi
  step "interp-1p5b" env CUDA_VISIBLE_DEVICES=0 \
    python -m reasoncontrol.stages.interp --config $CFG
  touch runs/.ap_gpu0_done
  log "GPU0 CHAIN DONE"
}

gpu1_chain(){
  waitpid "$WAIT1"
  # Finish the exit_only grid (banked points skip instantly by policy hash).
  for TAU in 0.7 0.8 0.9 0.95; do for K in 1 2; do
    step "exit-dev[t$TAU k$K]" env RC_SPLIT=dev CUDA_VISIBLE_DEVICES=1 \
      python -m reasoncontrol.stages.run_controller --config $CFG \
      --set policy.kind=exit_only --set policy.tau_exit=$TAU \
      --set policy.patience_k=$K --set datasets='[math_train,gsm8k]'
  done; done
  # Same-engine noop reference (min_chunks=5 keeps its hash off the vLLM noop).
  step "noop-dev" env RC_SPLIT=dev CUDA_VISIBLE_DEVICES=1 \
    python -m reasoncontrol.stages.run_controller --config $CFG \
    --set policy.kind=noop --set policy.min_chunks=5 \
    --set datasets='[math_train,gsm8k]'
  step "select" python scripts/select_operating_points.py \
    --families exit_only,static_budget,budget_prompt
  step "closed-loop-audit" python scripts/closed_loop_audit.py
  snap "GPU1 dev half done: exit grid + HF-noop dev + selection + audit"
  touch runs/.ap_gpu1_dev_done

  local T K BS BP
  T=$(selget exit_only tau_exit); K=$(selget exit_only patience_k)
  BS=$(selget static_budget budget); BP=$(selget budget_prompt budget)
  if [ -n "$T" ] && [ -n "$K" ]; then
    for S in 0 1 2 3 4 5 6 7; do
      ctl_test 1 "$CFG" $S 32 --set policy.kind=exit_only \
        --set policy.tau_exit=$T --set policy.patience_k=$K
    done
    snap "1.5B exit_only test @ tau=$T k=$K (8 seeds)"
  else
    log "STEPFAIL exit-test: no exit_only selection"
    echo "$(date -u +%FT%TZ) exit-test no-selection" >> runs/AUTOPILOT_FAILURES
  fi
  for S in 0 1 2 3 4 5 6 7; do
    [ -n "$BS" ] && base_test 1 "$CFG" $S \
      --set policy.kind=static_budget --set policy.budget=$BS
    [ -n "$BP" ] && base_test 1 "$CFG" $S \
      --set policy.kind=budget_prompt --set policy.budget=$BP
    base_test 1 "$CFG" $S --set policy.kind=noop
    base_test 1 "$CFG" $S --set policy.kind=concise_prompt
    base_test 1 "$CFG" $S --set policy.kind=trial_decode
  done
  snap "1.5B vLLM baseline test suite (8 seeds)"

  # 7B references: vLLM noop + selected static budget (4 seeds), one HF-noop
  # seed for the engine-gap estimate (documented trims).
  for S in 0 1 2 3; do
    base_test 1 "$CFG7" $S --set policy.kind=noop
    [ -n "$BS" ] && base_test 1 "$CFG7" $S \
      --set policy.kind=static_budget --set policy.budget=$BS
  done
  ctl_test 1 "$CFG7" 0 16 --set policy.kind=noop --set policy.min_chunks=5
  snap "7B test references (vLLM noop/static 4 seeds + 1 HF-noop seed)"
  step "interp-7b" env CUDA_VISIBLE_DEVICES=1 \
    python -m reasoncontrol.stages.interp --config $CFG7
  touch runs/.ap_gpu1_done
  log "GPU1 CHAIN DONE"
}

log "AUTOPILOT START (wait0=$WAIT0 wait1=$WAIT1)"
gpu0_chain >> runs/logs/autopilot_gpu0.log 2>&1 &
P0=$!
gpu1_chain >> runs/logs/autopilot_gpu1.log 2>&1 &
P1=$!
log "chains launched: gpu0=$P0 gpu1=$P1"
wait $P0 $P1

step "analyze-dev" env RC_SPLIT=dev python -m reasoncontrol.stages.analyze --config $CFG
step "analyze-test" env RC_SPLIT=test python -m reasoncontrol.stages.analyze --config $CFG
step "analyze-7b-test" env RC_SPLIT=test python -m reasoncontrol.stages.analyze --config $CFG7
NF=$(cat runs/AUTOPILOT_FAILURES 2>/dev/null | wc -l)
{ echo "# ReasonControl: ALL EXPERIMENTS DONE"
  echo
  echo "- Finished (UTC): $(date -u +%FT%TZ)"
  echo "- Step failures logged: $NF (see docs/AUTOLOG.md)"
  echo "- Safe to terminate the GPU instance AFTER verifying this commit is on GitHub main."
} > EXPERIMENTS_DONE.md
date -u > runs/AUTOPILOT_DONE
snap "AUTOPILOT COMPLETE (failures=$NF) — safe to terminate GPU instance"
log "AUTOPILOT COMPLETE failures=$NF"
