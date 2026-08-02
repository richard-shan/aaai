# ReasonControl: ALL EXPERIMENTS DONE

- Finished (UTC): 2026-08-02T09:17:41Z
- Step failures: 1 real (steering-acceptance crash, 2026-07-31), resolved by
  hand the same day — verdict REJECTED, D6 exit-led. The autopilot prints
  "failures=2" because it counts LINES in runs/AUTOPILOT_FAILURES and the
  resolution note is a second line; there was never a second failure.
- PRE-REGISTERED PRIMARY ENDPOINT FAILED its accuracy leg: token superiority
  passed (-52.4% think tokens) but the accuracy drop -0.039 [-0.054, -0.024]
  exceeds the 0.02 non-inferiority margin. See docs/RESULTS.md "PRIMARY
  ENDPOINT — FINAL". stages/analyze.py never computed this (it only emits the
  hierarchical test when a `full` steer+exit family is present on test, and
  steering was rejected), so it is computed by scripts/primary_endpoint.py.
- Test point: exit_only tau=0.7 k=2; answer budget 2048 tokens.
- Numbers regraded with the fixed extractor (docs/RESULTS.md).
- Safe to terminate the GPU instance AFTER verifying this commit is on GitHub main.
