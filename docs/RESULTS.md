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

## ⚠ MEASUREMENT DEFECT FOUND 2026-07-30 — everything below the "dev picture"
## heading is SUPERSEDED by the corrected tables in this section

Two grading defects were found while auditing a below-chance GPQA number
(0.177 and 0.121 vs 0.25 MCQ chance). Both are measurement-only: no rollout
was regenerated, and the fix is an offline re-grade of stored text
(`scripts/regrade.py` -> `runs/<model>/analysis/regrade_{records.parquet,
summary.csv}`). Originals are never modified.

1. **Answer-extraction miss.** `extract_answer` required `\boxed{}`,
   "answer is X", or `(A)`. R1-distill emits "**Answer:** X" / "Answer: C"
   in ~20-25% of completions (~60% on GPQA). Those were scored INCORRECT.
   Fixed in `data/grading.py` (strict tier = boxed / "answer is" / explicit
   answer marker; opt-in permissive tier = last expression on the final line,
   for completions truncated before committing). 51/51 tests pass, including
   regression tests that boxed and "answer is" behaviour is unchanged.
2. **Asymmetric answer budget (the "engine gap" was an artifact).** The HF
   controller loop stops at `gen.max_answer_tokens`=512 (`loop.py:430`) while
   the vLLM baseline path passes `max_think + max_answer` as ONE joint budget
   (`baselines.py:45`), so baselines could write thousands of answer tokens
   while the controller was cut at 512. Verbose conditions (noop, whose
   post-`</think>` recap is long) lost their answers to truncation.
   **With the 512 budget enforced on both engines, HF-noop = 0.703 and
   vLLM-noop = 0.704 on MATH dev — the engines agree to 0.001.** The
   "0.16 engine gap" reported earlier on Jul 30 is RETRACTED; it was entirely
   answer-budget truncation. Cross-engine numerics are not a confound.
   Note the controller's exit suffix is only `"\n</think>\n\n"` (loop.py:166)
   — exit_only gets no privileged answer elicitation, so the prompting is
   fair; noop truncates more because longer traces produce longer recaps.

**Corrected MATH-train dev (matched 512-token answer budget, fixed
extractor) — acc / mean think tokens:**

| condition | as-run (buggy) | corrected | think |
|---|---|---|---|
| exit_only tau.7/K2 | 0.682 | **0.812** | 2260 |
| exit_only tau.7/K1 | 0.660 | **0.777** | 1707 |
| static_budget B=4096 | 0.731 | 0.739 | 2593 |
| vLLM noop | 0.865 | 0.704 | 4529 |
| HF noop | 0.701 | 0.703 | 4470 |
| full (exit+steer) | 0.482 | 0.588 | 5265 |

GSM8K dev corrects even more sharply: exit_only 0.475-0.680 -> 0.835-0.890
(HF-noop 0.940 unchanged). **The earlier claim "exit_only loses 0.26-0.46 acc
on GSM8K" is RETRACTED — it was extraction, not policy.**

**Corrected 1-SE selection (pooled MATH+GSM8K dev, matched budget):**
exit_only picks **079322feb6 = tau*=0.7, K*=2 (pooled 0.817 @ 1909 tokens)**,
NOT the tau0.7/K1 point chosen under buggy grading; static_budget picks
974b1c9039 (B=2048, 0.772 @ 1588); budget_prompt 0.734 @ 3776 (it barely
restrains thinking, so it is an accuracy control, not an efficiency baseline).
Pooled noop at matched budget ~0.750 @ ~3860.

### Answer-budget control (2026-07-31) — RESOLVES the budget question

Re-ran the noop dev reference at `gen.max_answer_tokens=2048` (hash
5391bd1826; the 512 run is 888caea894):

| noop, MATH-train dev | acc | mean think |
|---|---|---|
| HF loop, 512-token answers | 0.703 | 4470 |
| **HF loop, 2048-token answers** | **0.855** | 4663 |
| vLLM, joint (effectively unlimited) answers | 0.865 | 4529 |

The 512 cap was costing noop **0.152 accuracy**. At an adequate budget the two
engines agree to 0.010 (0.855 vs 0.865) — a second, independent confirmation
that there is no cross-engine accuracy gap, only a budget-enforcement bug.

**This retracts the "exit_only dominates noop" reading** from the
512-matched-budget tables above. Those tables are internally consistent (the
same cap on both engines) but they systematically understate every condition
whose answers run long — which is exactly the control condition. The 2048
budget is therefore the primary, and the 512 tables become a robustness
appendix. Expected corrected relationship, pending exit_only@2048 (running):
exit_only ~0.81-0.84 @ ~2300 vs noop 0.855 @ 4663, i.e. a SMALL accuracy cost
for ~50% fewer think tokens — while still clearly beating static_budget
(0.740 @ 2593), whose vLLM baselines were never budget-limited.

Known limitation to state in the paper: the operating point was selected on
512-budget dev data (re-grading fixed the extractor but cannot un-truncate an
answer). Re-selecting the whole grid at 2048 costs ~28 GPU-hours and was not
affordable; instead the selected point is re-run at 2048 as a verification,
and the selection's sensitivity to the cap is analysed below.

#### Would re-selecting at 2048 change the operating point? No. (2026-08-02)

Estimated each grid point's 2048 accuracy from its unparsed-answer rate (the
truncation-damage proxy), calibrated on the two points measured at BOTH
budgets: noop 512->2048 recovered 0.2075 unparsed for +0.153 acc (0.74 per
unparsed answer), exit_only@selected recovered 0.0537 for +0.015 (0.28 per).

| point | think | acc@512 | unparsed | est. acc@2048 (0.28 / 0.74) |
|---|---|---|---|---|
| cbb8895c16 (t0.7/K1) | 1707 | 0.778 | 0.111 | 0.800 / 0.838 |
| **079322feb6 (t0.7/K2, SELECTED)** | 2260 | 0.813 | 0.084 | **0.828 / 0.852** |
| fdec51004e | 2214 | 0.764 | 0.113 | 0.787 / 0.825 |
| ab2ad2aa91 | 2752 | 0.784 | 0.116 | 0.808 / 0.848 |
| 7ea0180ba8 | 2783 | 0.769 | 0.120 | 0.794 / 0.835 |
| 42c3453b67 | 3353 | 0.774 | 0.129 | 0.801 / 0.847 |
| 5733e5f65d | 3387 | 0.759 | 0.152 | 0.793 / 0.849 |
| 7b873f923e | 3960 | 0.714 | 0.190 | 0.759 / 0.832 |

Under EITHER calibration the 1-SE rule (SE ~ 0.0133) re-selects
**079322feb6** — it has the highest estimated accuracy and nearly the fewest
tokens. The conservative calibration also predicts the one point actually
measured at 2048 to within 0.0001 (0.8276 est vs 0.8275 measured).

Two consequences worth stating in the paper:
1. The 512 cap was biased **toward** the selected point, not against it: it
   punished the high-token settings hardest (19.0% unparsed at 3960 think vs
   8.4% at 2260). Correcting it pushes toward MORE aggressive exits, i.e.
   away from closing the accuracy gap.
2. The gap is structural, not a tuning error. The best estimated accuracy
   anywhere on the grid at 2048 is ~0.83 vs noop 0.855 — the original
   "flat in tau" result surviving every correction. No threshold clears the
   -0.02 margin while still saving meaningful compute; the conservative end
   costs +78% tokens for perhaps +0.005 accuracy.

So re-running selection is NOT the fix. The binding constraint is the probe's
off-policy overconfidence (closed-loop audit: exits at claimed p 0.83-0.95,
realized 0.65-0.70) — a distribution-shift problem, since the probe is trained
on unintervened traces and then evaluated on states its own interventions
create. The indicated next step is iterative on-policy probe retraining
(DAgger-style), which is new method work, not a re-run.

**Headline depends on the answer budget, so the budget must be fixed:**
- At the matched 512-token budget (as pre-registered), exit_only *dominates*:
  0.817 @ 1909 vs noop 0.750 @ 3860 and static_budget 0.772 @ 1588.
- At an adequate budget (no truncation), noop rises to ~0.884 pooled
  (vLLM, unconstrained answers) while exit_only, which truncates in ~10% of
  MATH rollouts, would rise less — plausibly ~0.83. Then exit_only trades
  ~0.05 accuracy for ~50-60% fewer think tokens and still Pareto-dominates
  static_budget.
A 512-token cap that truncates ~20% of the *control's* answers is a fatal
review comment either way, so the remaining runs move to
`gen.max_answer_tokens=2048` (above the observed p99 answer length ~900
tokens, so NEITHER engine truncates and the joint-budget asymmetry becomes
moot without changing baseline code). The 512-budget dev sweep is retained
as a robustness appendix.

**Consequence for the in-flight test phase:** it was running the
buggy-grading operating point (tau0.7/K1) at the 512 budget, so it is being
restarted at the corrected point (tau0.7/K2) with the 2048 budget. Selection
remains dev-only; the test split is still evaluated once per selected point.

#### Paired 2048-budget dev comparison — COMPLETE (2026-07-31 08:28 UTC)

`exit_only` at the selected point (tau=0.7, K=2) re-ran at 2048 (hash
84bc53aa11), giving the other half of the control. Same engine (HF loop),
same budget, same problems:

| MATH-train dev | acc | mean think | | GSM8K dev | acc | mean think |
|---|---|---|---|---|---|---|
| noop @2048 | 0.855 | 4663 | | noop @2048 | 0.920 | 1913 |
| exit_only @2048 | 0.828 | 2223 | | exit_only @2048 | 0.825 | 501 |
| delta | **-0.027** | **-52%** | | delta | **-0.095** | **-74%** |

Pooled (800 MATH + 200 GSM8K dev rollouts): **exit_only 0.827 @ 1879 vs noop
0.868 @ 4113 — -0.041 accuracy for -54% think tokens.**

This is the honest headline and it lands between the two earlier readings:
early exit is *not* free (the 512-budget tables that showed it dominating were
an artifact), but it is also *not* the disaster the buggy extractor suggested.
The dataset split from the 512 analysis survives correction: the cost is small
on long MATH traces (-0.027) and concentrated on short GSM8K traces (-0.095),
where the settle-detector's off-policy overconfidence (closed-loop audit)
fires early on problems the model would have finished anyway.

The 512-budget exit_only run at this same point, re-graded at a matched
budget, gives 0.812 @ 2260 (MATH) / 0.835 @ 506 (GSM8K) — within noise of the
2048 run, confirming exit_only's answers rarely hit the cap and that the
selection made on 512 data transfers to the 2048 regime.

### Steering acceptance + D6 — DECIDED (2026-08-01, dev only)

`scripts/steering_acceptance.py` (re-graded with the fixed extractor at a
matched 512-token answer budget on both arms; reference = HF-loop noop dev,
hash 5391bd1826, 1000 rollouts):

| steer_only alpha=3 vs HF-noop, 250 paired dev problems / 1000 rollouts | value |
|---|---|
| paired accuracy delta (steered - unsteered) | **-0.215** [-0.251, -0.180] |
| max plausible accuracy drop (pre-registered margin 0.02) | **0.251** |
| paired think-token delta | **+3583** [+3131, +4063] |
| **acceptance** | **REJECTED** |

Steering fails the gate by an order of magnitude, and it *costs* tokens rather
than saving them, so the judge-verified phase-shift check is not run (it is
gated on acceptance passing). The regex paragraph screen is reported for
completeness and is itself the wrong sign: +alpha *lowered* the backtracking
rate (0.163 vs 0.231, delta -0.068 [-0.074, -0.062]) while nudging deduction
up (+0.026) and verification up (+0.007).

**D6 (pre-registered) fires: EXIT-LED HEADLINE.** Both arms of the rule agree
— acceptance failed, *and* steering's marginal token savings over exit-only
are negative (-2774 tokens, ratio -1.26, threshold <0.05). Steering therefore
ships as an honest negative result with a mechanism (the closed-loop
overconfidence audit), not as a contribution. No steering condition runs on
test.

**1.5B dev picture as of Jul 30 (SUPERSEDED — retained for provenance):**

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

## Test (8 seeds, single-shot at selected points)

### ★ PRIMARY ENDPOINT — FINAL (2026-08-02)

Pre-registered: pooled MATH-500 + GSM8K test, hierarchical (token superiority
-> accuracy non-inferiority at margin 0.02), cluster bootstrap 10k over
problems, against the SAME-ENGINE (HF-loop) noop reference.
`runs/analysis/<tag>/primary_endpoint.json`, via `scripts/primary_endpoint.py`.

| pooled MATH-500+GSM8K test | acc [95% CI] | mean think |
|---|---|---|
| **1.5B** exit_only t0.7/K2 (8 seeds) | 0.8165 [0.7960, 0.8362] | 1773 |
| 1.5B noop, same engine (3 seeds) | 0.8556 [0.8347, 0.8756] | 3722 |
| 1.5B noop, vLLM (8 seeds) | 0.8472 [0.8268, 0.8665] | 3796 |
| 1.5B static_budget B*=2048 (8 seeds) | 0.8028 [0.7797, 0.8252] | 1578 |
| **7B** exit_only, transferred (4 seeds) | 0.8627 [0.8427, 0.8817] | 1093 |
| 7B noop, same engine (1 seed) | 0.9360 [0.9173, 0.9533] | 2622 |
| 7B noop, vLLM (4 seeds) | 0.9233 [0.9070, 0.9387] | 2838 |
| 7B static_budget B*=2048 (4 seeds) | 0.8487 [0.8263, 0.8703] | 1499 |

**Verdict on the pre-registered hierarchy — the primary endpoint FAILS.**

| 1.5B, exit_only vs same-engine noop | result |
|---|---|
| gate 1: token superiority | **PASSED** — -1949 tokens [-2107, -1798], **-52.4%** |
| gate 2: accuracy non-inferiority (margin 0.02) | **FAILED** — delta -0.0391 [-0.0535, -0.0244] |

The lower CI bound of the accuracy drop is -0.054, well past the -0.02 margin,
so the controller does **not** clear its own pre-registered non-inferiority
bar. The same holds at 7B (-0.0733 [-0.0907, -0.0563]) and cross-engine
(-0.0307 [-0.0420, -0.0192]). This is stated first because it is the
pre-registered test and it did not pass; every favourable result below is
secondary to it.

**Where the controller does win: against the compute-matched baseline, at 7B.**

| exit_only vs static_budget B*=2048 | 1.5B | 7B |
|---|---|---|
| token superiority | **FAILED** (+195 tokens [+16, +387]) | **PASSED** (-406 [-499, -301], -27.1%) |
| accuracy delta | +0.0137 [+0.0005, +0.0277] | +0.0140 [+0.0000, +0.0287] |
| joint verdict | no dominance | **dominates** (cheaper AND non-inferior) |

At 1.5B the learned controller is *not* better than simply capping the budget:
it buys +0.014 accuracy while spending 12% MORE tokens. At 7B it dominates —
27% cheaper at equal-or-better accuracy. So the defensible claim is narrow and
scale-dependent: a probe-driven exit beats a fixed budget only once the model
is strong enough that *when* to stop varies meaningfully across problems.

Honest summary for the paper: **early exit buys a large, reliable compute
saving (~52-58% of think tokens) at an accuracy cost that is small but
statistically real, and larger than we pre-registered as acceptable.** It is
not a free lunch. Its one clean win over a compute-matched baseline is at 7B.

### 1.5B — LANDED (2026-08-01; 8 seeds each, answer budget 2048)

Mean over seeds +- SE of the seed mean. `exit_only` runs on the HF controller
loop; all baselines run on vLLM, so the cross-engine caveat applies to every
row-vs-row comparison here — **the within-engine comparison is exit_only vs
the HF-noop reference (3 seeds, in flight) and it is the primary endpoint.**

| policy | MATH-500 | GSM8K | GPQA-D | AIME'25 |
|---|---|---|---|---|
| noop (vLLM) | 0.833±0.002 @4653 | 0.877±0.005 @2082 | 0.240±0.008 @8027 | 0.225±0.016 @14604 |
| **exit_only t0.7/K2 (HF)** | **0.807±0.004 @2312** | **0.836±0.008 @694** | **0.323±0.016 @6538** | **0.237±0.020 @11492** |
| static_budget B*=2048 | 0.774±0.004 @1726 | 0.861±0.006 @1281 | 0.316±0.010 @1923 | 0.146±0.015 @2033 |
| budget_prompt B*=1024 | 0.832±0.002 @4514 | 0.867±0.007 @1916 | 0.253±0.011 @8015 | 0.258±0.022 @14069 |
| concise_prompt | 0.832±0.004 @3447 | 0.765±0.003 @344 | 0.296±0.009 @6693 | 0.254±0.021 @14090 |
| trial_decode | 0.493±0.003 @317 | 0.584±0.009 @301 | 0.135±0.006 @376 | 0.042±0.008 @325 |

Pooled primary endpoint (MATH-500 + GSM8K, rollout-weighted 500:250):
exit_only **0.817 @ 1773** vs vLLM-noop 0.848 @ 3796 (-0.031 acc, **-53%
tokens**) vs static_budget 0.803 @ 1578 (+0.014 acc, +12% tokens).

Reading it honestly: against the compute-matched baseline that actually
restrains tokens (static_budget), exit_only is **not** a clean Pareto win on
test — it buys +0.014 accuracy for 12% more tokens, and static_budget wins
outright on GSM8K. The unambiguous wins are (a) the ~53% token cut vs noop for
-0.031 accuracy, and (b) GPQA, where exit_only beats every baseline
(0.323 vs 0.240 noop) — an out-of-distribution transfer the probe was never
tuned on. Cluster-bootstrap CIs and the hierarchical token-superiority ->
accuracy-non-inferiority test replace these seed-mean SEs once the analyze
step runs; the HF-noop reference decides the within-engine claim.

Retained robustness datapoint: the 2 v1 seeds at the *uncorrected* point
(t0.7/K1, 512 budget, hash cbb8895c16) score 0.628/0.654/0.149/0.150 — the
gap to the corrected point is almost entirely the 512-token answer cap, and
they are reported only as provenance, never as an endpoint.

- [x] HF-noop test reference (3 seeds) — within-engine primary, see above
- [x] 7B transfer at 1.5B-selected points (4 seeds + refs + 1 within-engine noop)

### 7B transfer — LANDED (2026-08-02; 4 seeds, answer budget 2048)

The 1.5B-selected operating point applied to the 7B with NO retuning.

| policy | MATH-500 | GSM8K | GPQA-D | AIME'25 |
|---|---|---|---|---|
| noop (vLLM, 4 seeds) | 0.924 @3548 | 0.923 @1417 | 0.381 @6432 | 0.392 @13179 |
| noop (HF, 1 seed) | 0.940 @3240 | 0.928 @1387 | 0.384 @6573 | 0.333 @12450 |
| **exit_only (HF, 4 seeds)** | 0.846 @1398 | 0.897 @483 | **0.433 @2607** | 0.258 @7805 |
| static_budget B*=2048 | 0.815 @1682 | 0.916 @1134 | 0.405 @1916 | 0.217 @2039 |

Cost of early exit scales with how much genuine search the problem needs:
GPQA **+0.049** (exit wins), GSM8K -0.031, MATH -0.094, AIME -0.075 (n=30,
descriptive anchor only). The accuracy cost at 7B is ~2x the 1.5B cost
(-0.073 vs -0.039 pooled) for a comparable token saving — a threshold tuned on
a weaker model exits too eagerly for a stronger one. Selection at 7B was not
affordable and is left as stated future work; this is a transfer result, not a
tuned one.

### Cross-engine agreement — the "engine gap" is fully closed

The 0.16 "engine gap" reported on 2026-07-30 was an artifact of the answer
budget defect. At 2048 tokens the two engines agree on every dataset at both
scales:

| noop, test | 1.5B HF / vLLM | 7B HF / vLLM |
|---|---|---|
| MATH-500 | 0.844 / 0.833 | 0.940 / 0.924 |
| GSM8K | 0.879 / 0.877 | 0.928 / 0.923 |
| GPQA-D | 0.249 / 0.240 | 0.384 / 0.381 |
| AIME'25 | 0.244 / 0.225 | 0.333 / 0.392 |
| **pooled primary** | **0.856 / 0.847** | **0.936 / 0.923** |

### Interp (logit lens) — supports the phase probes, NOT v_conv

`runs/analysis/<tag>/interp.json`. The phase directions decode to exactly the
concepts they are named for, including across languages:
- `backtracking_L21` promotes ` but`, ` However`, `But`, `但是`, `然而`, ` perhaps`
- `verification_L21` promotes ` checking`, ` check`, ` verification`, ` confirming`

But `v_conv` — the direction that actually drives the EXIT decision — does
not: it promotes ` Norton`, `vit`, ` whatsoever`, `ser`, `om`, with
`<|end_of_sentence|>` the single meaningful entry. Reported as a negative:
the interpretability claim is supported for the phase probes and unsupported
for the convergence probe, which is consistent with the closed-loop audit
finding that the settle-detector is well-calibrated on-policy yet badly
overconfident under intervention.
- [ ] Primary endpoint: pooled MATH-500+GSM8K, hierarchical (token
      superiority -> accuracy non-inferiority), cluster bootstrap 10k
- [ ] AIME'25 + GPQA descriptive CIs (anchors)
- [x] Steering acceptance (REJECTED) + pre-registered D6 decision (EXIT-LED)
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
12. **Grading defect fixed mid-run (2026-07-30, pre-test-completion).**
    `extract_answer` missed the "**Answer:** X" form; all stored rollouts were
    re-graded offline from stored text (`scripts/regrade.py`). No rollout was
    regenerated and no original file was modified. The strict tier (boxed /
    "answer is" / explicit answer marker) is primary; a permissive
    last-expression tier is reported only as a robustness check.
13. **Answer budget raised 512 -> 2048 for all runs after 22:12 UTC Jul 30.**
    The 512 cap was enforced on the HF controller but not on the vLLM
    baselines, truncating verbose conditions' answers. 2048 exceeds the
    observed p99 answer length (~900 tokens) so neither engine truncates.
    Dev sweeps stay at 512 (re-graded with the budget enforced on both engines
    for a matched comparison); a 2048 dev control re-runs noop and
    exit_only@selected so the cap's effect is measured, not assumed.
14. **Operating point re-selected on corrected grades** (dev-only, same 1-SE
    rule): exit_only tau*=0.7 K*=2 (was K*=1), static_budget B*=2048 (was
    4096). The two v1 test seeds at the old point (hash cbb8895c16, 512
    budget) are retained as a robustness datapoint, not the primary endpoint.
15. steer_only dev sweep trimmed to alpha=3 (from {3,6,9}): with corrected
    grading the full policy is 0.59-0.64 vs exit_only 0.78-0.82 on MATH dev,
    so the remaining alphas are futile; alpha=3 still carries the
    pre-registered paired acceptance test.
17. **Steering acceptance was re-graded, not taken from stored labels.** The
    `correct` field in the dev result files predates the extractor fix, so
    `scripts/steering_acceptance.py` now re-grades both arms in-process with
    the fixed extractor at a matched 512-token answer budget, and records the
    noop reference hash it used in the output JSON. (It also fixed a crash:
    the noop reference was unpacked as `(policy, df)` instead of `df`, which
    made the gate fail with an AttributeError on its first scheduled run.)
18. **Autopilot snapshot commits were silently dropping artifacts.**
    `git add a b missing` aborts and stages *nothing*; the snapshot listed
    `EXPERIMENTS_DONE.md` and `runs/r1_qwen_7b/analysis` before they existed,
    so between Jul 31 08:32 and Aug 1 02:30 UTC every milestone logged
    "committed+pushed" while committing nothing. Paths are now added
    individually and a failed commit is reported instead of swallowed. No
    experimental data was lost (result files are written directly to disk).
16. Steering acceptance: regex phase-rate shift computed on paragraph-split
    chunks (approximation) as a screen; the pre-registered judge-verified
    shift is required (and will be run) only if the paired accuracy gate
    passes. Full policy runs on test ONLY if acceptance passes.
