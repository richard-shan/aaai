#!/usr/bin/env bash
# Regenerate docs/AUTOLOG.md (machine-generated results record) and commit+push
# every durable artifact. Called by autopilot.sh at each milestone; safe to run
# by hand. Serialized via flock (both GPU chains call it).
set -u
cd /lambda/nfs/aaai/aaai
MSG=${1:-autopilot snapshot}

exec 9> runs/.snapshot.lock
flock 9

{
  echo "# Autopilot auto-log (machine-generated; prose lives in docs/RESULTS.md)"
  echo
  echo "Updated: $(date -u +%FT%TZ) — last milestone: ${MSG}"
  echo
  echo "## Step failures"
  echo '```'
  cat runs/AUTOPILOT_FAILURES 2>/dev/null || echo "(none)"
  echo '```'
  echo
  echo "## Banked results (all acc= lines, all logs, deduped)"
  echo '```'
  grep -h "acc=" runs/logs/*.log 2>/dev/null | grep -v "rollouts (" | sort -u
  echo '```'
  for f in runs/*/analysis/selection_dev.json \
           runs/*/analysis/steering_acceptance.json \
           runs/*/analysis/closed_loop_audit.json; do
    [ -e "$f" ] && { echo; echo "## $f"; echo '```json'; cat "$f"; echo '```'; }
  done
  for f in runs/*/analysis/summary_*.csv; do
    [ -e "$f" ] && { echo; echo "## $f"; echo '```'; cat "$f"; echo '```'; }
  done
} > docs/AUTOLOG.md

# NOTE: `git add a b missing` aborts and stages NOTHING. Add each path
# separately so one absent artifact cannot silently skip the whole commit.
for p in docs/AUTOLOG.md STATUS.md docs/RESULTS.md EXPERIMENTS_DONE.md; do
  [ -e "$p" ] && git add "$p" 2>/dev/null
done
# runs/analysis/<tag>/ is where the pipeline's own stages write (interp.json);
# runs/<tag>/analysis/ is where the offline scripts write. Both must be saved:
# runs/ is gitignored, so anything not force-added here dies with the instance.
for p in runs/r1_qwen_1p5b/analysis runs/r1_qwen_7b/analysis \
         runs/analysis \
         runs/AUTOPILOT_FAILURES runs/AUTOPILOT_DONE; do
  [ -e "$p" ] && git add -f "$p" 2>/dev/null
done
if git diff --cached --quiet; then
  echo "[snapshot $(date -u +%FT%TZ)] nothing to commit: ${MSG}"
else
  if git commit -m "autopilot: ${MSG}" >/dev/null 2>&1; then
    git push origin main >/dev/null 2>&1 || { sleep 30; git push origin main >/dev/null 2>&1; }
    echo "[snapshot $(date -u +%FT%TZ)] committed+pushed: ${MSG}"
  else
    echo "[snapshot $(date -u +%FT%TZ)] COMMIT FAILED: ${MSG}"
  fi
fi
