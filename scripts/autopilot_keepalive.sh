#!/usr/bin/env bash
# Restart the autopilot if it dies, without an agent in the loop.
# Runs from cron every 10 min alongside notify_done.sh.
#
# Deliberately conservative: it relaunches ONLY when the session leader is
# gone AND no stage process is alive AND the run is not finished. autopilot2.sh
# is idempotent (it skips result files that already exist), so a spurious
# relaunch wastes time but cannot corrupt banked results; a spurious DOUBLE
# launch could contend for GPU memory, hence the flock and the strict checks.
set -u
REPO=/lambda/nfs/aaai/aaai
cd "$REPO" || exit 0
LOG=runs/logs/keepalive.log
mark(){ echo "[keepalive $(date -u +%FT%TZ)] $*" >> "$LOG"; }

[ -f runs/AUTOPILOT_DONE ] && exit 0   # finished; nothing to keep alive

exec 7> runs/.keepalive.lock
flock -n 7 || exit 0

pgrep -f "bash scripts/autopilot2\.sh" >/dev/null && exit 0   # leader alive
pgrep -f "reasoncontrol\.stages\.run_(controller|baselines)" >/dev/null && {
  mark "leader gone but a stage is still running; waiting for it to finish"
  exit 0
}

mark "autopilot2 leader and all stages are gone, AUTOPILOT_DONE absent — relaunching"
nohup setsid bash scripts/autopilot2.sh >> runs/logs/autopilot.log 2>&1 < /dev/null &
disown 2>/dev/null || true
sleep 5
if pgrep -f "bash scripts/autopilot2\.sh" >/dev/null; then
  mark "relaunch OK (pid $(pgrep -f 'bash scripts/autopilot2\.sh' | head -1))"
else
  mark "RELAUNCH FAILED — see runs/logs/autopilot.log"
fi
