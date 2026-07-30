# ReasonControl — running results log (paper-facing)

Convention: every number here is traceable to an artifact under `runs/` (noted
per section). Update this file as results land; STATUS.md tracks operations,
this file tracks science. All "dev" numbers are exploratory; test numbers are
single-shot at dev-selected operating points per the pre-registered protocol.

## Setup (fixed)

- Models: DeepSeek-R1-Distill-Qwen-1.5B (probe layer L15, steer L21) and
  -7B (probe L18, steer L21); fp16; HF loop for controller, vLLM 0.26 for
  rollouts/baselines; torch 2.11.0, transformers 5.14.1.
- Data splits (seed 0, fixed): MATH-train 500 probe-train + 200 dev; MATH-500
  all-500 test; GSM8K 300/50/250 probe/dev/test; AIME'24 dev (30) / AIME'25
  test (30, uncontaminated anchor); GPQA-diamond 198 test-only (MCQ, choices
  shuffled at seed 0; anchor, descriptive CIs only).
- Rollouts: temp 0.6, top-p 0.95, n=4 per problem (dev sweeps), 16k think cap
  (24k AIME), 512-token answers.

## H1 — Convergence is linearly decodable (D4 gate: **GO both models**)

Gate: conv AUC >= 0.75 on MATH dev AND >= 0.05 AUC margin over the
position-only shallow baseline in EVERY position stratum.
Artifacts: `runs/<model>/probes/conv_linear/L*.metrics.json`; gate line in
`runs/logs/run_{1p5b,7b}.log`.

- 1.5B: **AUC 0.882 (L15)**, min within-stratum margin **0.147** => GO
- 7B:   **AUC 0.854 (L18)**, min within-stratum margin **0.246** => GO
  (margin grows with scale — supports H1 not being a position artifact)

Per-layer AUC / acc / ECE (n = 176,448 boundaries 1.5B; 158,961 7B):

| layer | 1.5B AUC | 1.5B acc | 1.5B ECE | 7B AUC | 7B acc | 7B ECE |
|---|---|---|---|---|---|---|
| L9  | 0.857 | 0.783 | 0.018 | 0.838 | 0.762 | 0.060 |
| L12 | 0.873 | 0.802 | 0.020 | 0.847 | 0.773 | 0.054 |
| L15 | **0.882** | 0.809 | 0.022 | 0.850 | 0.776 | 0.055 |
| L18 | 0.880 | 0.808 | 0.019 | **0.854** | 0.779 | 0.055 |
| L21 | 0.876 | 0.803 | 0.022 | 0.850 | 0.776 | 0.054 |
| L24 | 0.869 | 0.796 | 0.026 | 0.845 | 0.771 | 0.060 |
| L26 | 0.867 | 0.792 | 0.024 | 0.840 | 0.767 | 0.057 |
| L27 | 0.864 | 0.790 | 0.025 | 0.835 | 0.763 | 0.060 |

Notes for the paper: clean mid-network peak at both scales (L15/28 ≈ 0.54
depth for 1.5B; L18/28 ≈ 0.64 for 7B); probes are well-calibrated at 1.5B
(ECE ≈ 0.02) and moderately at 7B (ECE ≈ 0.055) — cite when discussing
tau_exit thresholds transferring across scales.

Phase probes (multiclass, macro AUC): 1.5B L15 0.938, L18 0.932; 7B L18
0.937. Accuracy ≈ 0.50 over 5 classes. Secondary signal; kappa caveat below.

## Label quality

- Forced-answer convergence labels: parse rate 1.000 everywhere.
  Boundaries: 1.5B 113,136 (MATH-train) / 78,911 (MATH-500) / 63,312 (GSM8K)
  / 13,970 (AIME); 7B 105,153 / 73,977 / 53,808 / 13,678.
  Prefix-cache hit rates 94.4–95.1% (two-phase sharded submission).
- Phase labels: regex + Qwen2.5-32B-AWQ judge on stratified subsets.
  Cohen's kappa (regex vs judge): 1.5B 0.453/0.472/0.524/0.426
  (MATH-train/MATH-500/GSM8K/AIME); 7B 0.456/0.454/0.492/0.451.
  Moderate agreement — report as labeling-noise caveat on phase results;
  convergence labels (primary) are judge-independent.
- Cross-engine sanity: greedy argmax agreement vLLM->HF 0.972 (pilot; gate
  amended to >=0.95 with margin diagnostic — mismatches sit at low-margin
  positions, median margin 0.69 vs 3.7 at matches; benign fp16 numerics).
  Sampled-trace agreement 0.89–0.91 at temp 0.6 (expected).

## Steering vectors (diff-of-means, probe-orthogonalized)

Artifacts: `runs/<model>/steering/`, log lines `build_steering:`.
Cos(vector, conv-probe direction) before -> after orthogonalization:

- 1.5B L21: verification (n=4,855) 0.056 -> ~0; backtracking (n=24,789)
  0.013 -> ~0; deduction (n=57,762) -0.022 -> ~0
- 7B L21: verification (n=4,970) 0.027 -> ~0; backtracking (n=19,406)
  0.005 -> ~0; deduction (n=55,674) -0.011 -> ~0

## Dev sweeps (in progress)

Grid: full & exit_only x tau_exit {0.7,0.8,0.9,0.95} x patience {1,2}, n=4,
MATH-train dev + GSM8K dev; steer_only alpha {3,6,9}; noop reference; vLLM
baselines (static_budget / budget_prompt B {1024,2048,4096,8192},
concise_prompt, trial_decode). ~2.5–3 h per controller point (HF loop).

RESULTS: (fill from `RC_SPLIT=dev analyze` when both sweep halves land;
running machine-generated record: docs/AUTOLOG.md)

First banked points (full, tau=0.7, K=1, n=4 — acc / mean think tokens):

| model | MATH-train dev | GSM8K dev |
|---|---|---|
| 1.5B | 0.482 / 5,265 | 0.615 / 2,351 |
| 7B   | 0.606 / 2,045 | 0.840 / 717   |

(7B reaches higher accuracy at ~2.5-3x fewer think tokens under the same
aggressive exit point — consistent with better-calibrated confidence at scale.)

**1.5B dev picture as of Jul 30 (headline-relevant, mostly negative):**

- exit_only is FLAT in tau on MATH-train dev: 0.660@1707 (t.7K1),
  0.682@2260 (t.7K2), 0.627@2214 (t.8K1), 0.637@2752 (t.8K2), 0.646@2783
  (t.9K1), 0.635@3353 (t.9K2) — raising the exit threshold buys tokens but no
  accuracy, and sits BELOW static_budget (0.731@2593) and vLLM-noop
  (0.865@4529). Key caveat: controller numbers are HF-loop, baselines are
  vLLM (cross-engine sampled agreement ~0.9); the same-engine HF-noop dev
  reference decides how much of the gap is engine vs policy.
- **HF-noop dev reference landed (Jul 30): MATH-train dev acc=0.701 @ 4470
  mean think tokens** (hash 888caea894; GSM8K half still running). The
  HF-loop engine itself scores 0.164 BELOW vLLM-noop (0.701 vs 0.865) at
  essentially the same token budget — i.e. most of the exit_only-vs-baseline
  gap above is ENGINE, not policy. Same-engine comparison, MATH dev:
  exit_only t.7/K2 = 0.682@2260 vs HF-noop 0.701@4470 — **-0.019 acc for
  -49% think tokens**; t.7/K1 = 0.660@1707 — -0.041 acc for -62% tokens.
  This substantially revives the exit-led headline ON MATH (near-iso-accuracy
  early exit within-engine), makes the pre-registered same-engine test
  comparison (HF-noop 3 seeds vs exit_only 8 seeds) the primary within-engine
  check, and reframes much of the cross-engine gap as an implementation/
  deployment caveat. It also sharpens the closed-loop audit reading on MATH:
  "realized ~0.65-0.70 acc on exited rollouts" ≈ the HF engine's own noop
  ceiling (0.701). Cross-engine accuracy claims stay off the table per
  protocol; the paper's non-inferiority endpoint must be engine-matched.
- **BUT the engine gap is dataset-dependent — GSM8K tells the opposite story
  (HF-noop gsm8k/dev acc=0.940 @ 1417 vs vLLM-noop 0.960 @ 1308, gap only
  0.02):** same-engine, exit_only loses 0.26-0.46 acc on GSM8K (best point
  0.680@426 t.7K1 vs noop 0.940@1417). On short/easy traces where the base
  engine is fine, premature exit driven by the overconfident settle-detector
  genuinely destroys accuracy; on long MATH traces the HF engine itself is
  the binding constraint and exit is near-free. Honest summary for the paper:
  early exit is near-iso-accuracy at ~half the tokens on MATH (within
  engine), harmful on GSM8K, and the probe's off-policy overconfidence
  (closed-loop audit) is the mechanism for the GSM8K failure.
- **1-SE dev selection (pooled MATH+GSM8K dev, runs/r1_qwen_1p5b/analysis/
  selection_dev.json):** exit_only picks tau*=0.7, K*=1 (pooled 0.664@1451);
  static_budget B*=4096 (0.770@2319); budget_prompt B*=1024 (0.873@3776,
  pooled — note budget_prompt barely restrains tokens, it's accuracy-
  preserving but not a real efficiency baseline). Pooled same-engine
  HF-noop reference ≈ 0.749 @ ~3860. Test phase runs at these points.
- full (exit+steer) is flat-bad: 0.476-0.482 on MATH across tau — steering
  costs ~0.15 acc vs exit_only at matched tau AND inflates tokens. D6
  exit-led re-headline is the expected outcome (formal acceptance pending).
- **Closed-loop calibration audit** (runs/r1_qwen_1p5b/analysis/
  closed_loop_audit.json; probe = conv_matches_final, self-supervised):
  rollouts that exited did so at mean claimed p~0.83-0.95 but realized only
  ~0.65-0.70 accuracy — the settle-detector is severely overconfident under
  intervention despite on-policy ECE~0.02. Cap-hitting rollouts are near-zero
  accuracy (0.00-0.26; full-policy cap-hitters ~0.01, i.e. long-horizon
  steering destroys traces). This is the mechanistic explanation for the flat
  Pareto curve and a headline-grade honest-negative analysis for the paper.

- [ ] full-family dev Pareto (acc vs mean think tokens) table
- [ ] exit_only dev Pareto table
- [ ] steer_only dev effects + paired acceptance inputs
- [ ] baseline dev tables
- [ ] Selected operating points (1-SE rule, pooled dev): full=(tau*, K*),
      exit_only=(tau*, K*), budgets B*

## Test (8 seeds, single-shot at selected points) — PENDING

- [ ] noop (8 seeds, running early on GPU1)
- [ ] full @ selected; exit_only @ selected; baselines @ B*
- [ ] 7B transfer at 1.5B-selected points
- [ ] Primary endpoint: pooled MATH-500+GSM8K, hierarchical (token
      superiority -> accuracy non-inferiority), cluster bootstrap 10k
- [ ] AIME'25 + GPQA descriptive CIs (anchors)
- [ ] Steering acceptance + pre-registered D6 decision
- [ ] Interp: logit lens + SAE on probe/steer directions

## Deviations & amendments (all pre-test, all documented in STATUS.md)

1. Greedy cross-engine consistency gate 0.99 -> 0.95+margin-diagnostic
   (pre-launch; fp16 kernel numerics; README §6).
2. Sweep grid = plan's 16 configs (script had over-provisioned 24) split
   across GPUs; n=4 kept per plan.
3. 7B: no independent grid — transfer study at 1.5B-selected operating points
   (walltime; supports scale-transfer claim instead).
4. GPQA manifest built standalone post-launch (RNG stream differs from a
   monolithic prepare_data run; manifest is the frozen source of truth).
5. Stray offline GPQA rollouts exist (relaunch artifact); unused by any
   analysis.
6. HF-loop noop runs use `policy.min_chunks=5` (behaviorally inert for noop)
   solely to disambiguate the policy hash from the vLLM baseline noop
   (identical cfg -> identical hash -> silent skip of one engine's results).
7. Full-policy dev grid futility-trimmed 8 -> 5 points (0.7/0.8 x K1/K2 +
   0.9/K1): dev curve flat at 0.476-0.482 and strictly dominated; 0.9/K2 and
   both 0.95 points cut (walltime/billing).
8. HF-loop noop TEST reference trimmed 8 -> 3 seeds; it is an engine control
   only — the headline noop comparison uses the 8-seed vLLM runs.
9. 7B transfer scope: exit_only @ 1.5B-selected point x 4 seeds + vLLM
   noop/static_budget@B* x 4 seeds + 1 HF-noop seed (engine-gap estimate).
   Full 8-seed treatment reserved for the 1.5B primary endpoint.
10. AIME test runs execute in a separate invocation at the 24k think cap with
    reduced batch (32 for 1.5B, 16 for 7B — KV-cache memory); other test sets
    at the standard 16k cap, matching the plan's per-dataset caps.
11. Steering acceptance: regex phase-rate shift computed on paragraph-split
    chunks (approximation) as a screen; the pre-registered judge-verified
    shift is required (and will be run) only if the paired accuracy gate
    passes. Full policy runs on test ONLY if acceptance passes.
