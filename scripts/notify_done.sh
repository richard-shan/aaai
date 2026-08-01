#!/usr/bin/env bash
# Session-independent completion watchdog for the ReasonControl autopilot.
#
# Installed in the user's crontab (every 10 min) so it survives agent death,
# SSH drop, and reboot. Idempotent: does nothing until runs/AUTOPILOT_DONE
# exists, and at most one notification is ever sent (runs/.user_notified).
#
# On completion it, in order:
#   1. verifies the final commit is actually on origin/main (pushing if not),
#   2. writes and pushes SAFE_TO_TERMINATE.md — visible on GitHub with no
#      credentials beyond the existing deploy key,
#   3. emails ~/.smtp_creds:SMTP_TO if that file exists (see below),
#   4. touches runs/.user_notified so it never fires twice.
#
# To enable email, create ~/.smtp_creds with mode 0600:
#   SMTP_HOST=smtp.gmail.com
#   SMTP_PORT=587
#   SMTP_USER=you@gmail.com
#   SMTP_PASS=<16-char Google app password, NOT your account password>
#   SMTP_TO=you@gmail.com
# Without that file everything else still runs; the email step is skipped and
# logged. The file is never read into git and never echoed.
set -u
REPO=/lambda/nfs/aaai/aaai
cd "$REPO" || exit 0
LOG=runs/logs/notify_done.log
mark(){ echo "[notify_done $(date -u +%FT%TZ)] $*" >> "$LOG"; }

[ -f runs/AUTOPILOT_DONE ]   || exit 0
[ -f runs/.user_notified ]   && exit 0

exec 8> runs/.notify.lock
flock -n 8 || exit 0

mark "AUTOPILOT_DONE detected; starting notification"

# 1. make sure everything is on origin/main
git fetch -q origin 2>>"$LOG"
if [ -n "$(git log origin/main..main --oneline 2>/dev/null)" ] \
   || [ -n "$(git status --porcelain)" ]; then
  mark "unpushed work found; running snapshot"
  bash scripts/autopilot_snapshot.sh "completion: final push before notify" >>"$LOG" 2>&1
  git push origin main >>"$LOG" 2>&1
  git fetch -q origin 2>>"$LOG"
fi
HEAD_SHA=$(git rev-parse main)
ORIGIN_SHA=$(git rev-parse origin/main)
if [ "$HEAD_SHA" = "$ORIGIN_SHA" ]; then
  PUSH_STATE="all commits confirmed on origin/main ($HEAD_SHA)"
else
  PUSH_STATE="WARNING: local $HEAD_SHA != origin/main $ORIGIN_SHA — push failed, do NOT terminate yet"
fi
mark "$PUSH_STATE"

# 2. a marker file on GitHub, readable without any extra credentials
{
  echo "# SAFE TO TERMINATE THE GPU INSTANCE"
  echo
  echo "Completed: $(cat runs/AUTOPILOT_DONE 2>/dev/null)"
  echo "Notified:  $(date -u +%FT%TZ)"
  echo
  echo "Push state: ${PUSH_STATE}"
  echo
  echo "Step failures during the run:"
  echo '```'
  cat runs/AUTOPILOT_FAILURES 2>/dev/null || echo "(none)"
  echo '```'
  echo
  echo "Results: docs/RESULTS.md · machine log: docs/AUTOLOG.md · summary: EXPERIMENTS_DONE.md"
} > SAFE_TO_TERMINATE.md
git add SAFE_TO_TERMINATE.md 2>>"$LOG"
git commit -q -m "SAFE TO TERMINATE: all experiments complete and pushed" >>"$LOG" 2>&1
git push origin main >>"$LOG" 2>&1 || { sleep 30; git push origin main >>"$LOG" 2>&1; }

# 3. email, if credentials were provided
if [ -r "$HOME/.smtp_creds" ]; then
  SUBJ="ReasonControl: all experiments done — safe to terminate the GPU instance"
  BODY="$(printf '%s\n\n%s\n\n%s\n' \
    "All ReasonControl experiments finished at $(cat runs/AUTOPILOT_DONE 2>/dev/null)." \
    "${PUSH_STATE}" \
    "See SAFE_TO_TERMINATE.md and docs/RESULTS.md at github.com/richard-shan/aaai")"
  SUBJ="$SUBJ" BODY="$BODY" python3 - <<'PY' >>"$LOG" 2>&1
import os, smtplib, ssl
from email.message import EmailMessage
cfg = {}
with open(os.path.expanduser("~/.smtp_creds")) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
msg = EmailMessage()
msg["Subject"] = os.environ["SUBJ"]
msg["From"] = cfg["SMTP_USER"]
msg["To"] = cfg.get("SMTP_TO", cfg["SMTP_USER"])
msg.set_content(os.environ["BODY"])
with smtplib.SMTP(cfg.get("SMTP_HOST", "smtp.gmail.com"),
                  int(cfg.get("SMTP_PORT", 587)), timeout=60) as s:
    s.starttls(context=ssl.create_default_context())
    s.login(cfg["SMTP_USER"], cfg["SMTP_PASS"])
    s.send_message(msg)
print("email sent to", msg["To"])
PY
  mark "email step finished (rc=$?)"
else
  mark "no ~/.smtp_creds — email skipped; SAFE_TO_TERMINATE.md pushed to GitHub instead"
fi

# 4. never fire twice
date -u +%FT%TZ > runs/.user_notified
mark "done"
