# STATUS — handoff to GPU-box agent

**Project:** ReasonControl — interpretable latent-state controller for efficient
reasoning in R1-distill LLMs. AAAI target. Full research plan: `docs/PLAN.md`
(read it first — hypotheses H1–H3, baselines, statistics, risk register,
day-by-day schedule live there). Method README: `README.md`.

## Where things stand (as of 2026-07-28)

DONE (in this repo, commit history on `main`):
- Complete codebase for the 12-stage DAG (`src/reasoncontrol/stages/`): data
  prep, rollout generation, chunking, activation capture, forced-answer
  convergence labels, phase labels, probes, steering vectors, HF closed-loop
  controller, vLLM baselines, analysis, interp.
- 47 CPU unit tests, all passing (`make test`). Written and validated against
  `transformers==5.14.1` (5.x APIs: bare-Tensor layer outputs, StaticCache).
- Design decisions already stress-tested by an adversarial review: do NOT
  reintroduce `}`-stop-string answer parsing, string-equality grading, test-set
  sweeps, or probe evaluation without the position-controlled baseline —
  the rationale for each is in `docs/PLAN.md`.

NOT DONE (this container was CPU-only with huggingface.co blocked):
- `make smoke` / `make smoke-gpu` (first thing to run here)
- All real data generation, probe training, steering, controller runs
- Asset-ID verification on the hub (esp. the two AIME dataset ids and gated GPQA)

## Your immediate task list (in order)

1. `make setup-gpu` (let vllm drive the torch pin; if transformers resolves
   away from 5.14.x, run `make test` and fix any API drift before continuing).
2. Acceptance checklist in `README.md` — disk (>=250 GB incl. HF_HOME),
   `HF_TOKEN` with GPQA terms accepted, asset ids, `make test`, `make smoke`,
   `make smoke-gpu`.
3. Launch stage pipeline: `bash scripts/run_all_1p5b.sh` under
   `CUDA_VISIBLE_DEVICES=0`; same script with
   `--config configs/base.yaml configs/models/r1_qwen_7b.yaml` under
   `CUDA_VISIBLE_DEVICES=1`.
4. **D4 go/no-go gate** (printed by `train_probes`): conv-probe AUC >= 0.75 on
   MATH dev AND >= 0.05 AUC over the shallow baseline within every position
   stratum. If NO-GO, consult the risk register in `docs/PLAN.md` before
   changing anything.
5. Dev sweeps -> pick operating points (1-SE rule) -> test runs
   (`RC_SPLIT=test`, 8 seeds for headline conditions). Sweeps NEVER run on test.

## Hard guardrails (from the reviewed plan — do not bend)

- Grid/threshold selection on dev only; test evaluated once per selected point.
- AIME'25 + GPQA are the uncontaminated anchors for probe claims; their
  accuracy comparisons are descriptive CIs only (no non-inferiority claims).
- Steering ships only if the paired acceptance criterion passes
  (`steering/validate.py`: upper CI bound of accuracy drop < 2% on >= 200
  paired dev rollouts) AND the judge-verified phase-rate shift confirms
  causality (regex alone is circular).
- Pre-registered D6 decision: if steering adds < 5% of exit-only's token
  savings at matched accuracy, re-headline exit-led (see plan §Direction).

## Progress log (append below, newest first)

- 2026-07-28 (production, evening): both pipelines relaunched detached
  (nohup/setsid; logs in runs/logs/) after harness-tied background tasks were
  killed with the session (~2h lost, shard-resume worked as designed). 1.5B:
  generate/chunk/capture/forced_answer done for all 4 datasets (parse 1.000,
  cache hits 94.8-95.1%), label_phase kappa 0.43-0.52, and **D4 gate = GO**
  (`conv AUC=0.882 at L15; min within-stratum margin over shallow=0.147`) —
  both criteria pass (>=0.75 AUC, >=0.05 margin). Now in steering/sweeps.
  7B: resumed mid-forced_answer (math_train 105k boundaries done, parse 1.000,
  hit 94.8%); D4 pending. HF_TOKEN arrived -> stored at ~/.hf_token (0600);
  GPQA-diamond manifest built standalone (198 MCQ problems, choices shuffled;
  note: standalone build advances the seed-0 RNG differently than a full
  prepare_data pass would — the written manifest is the single source of truth
  and must not be regenerated). GPQA needs no offline stages (test-only anchor;
  controller/baselines roll out at eval time from the manifest).

- 2026-07-28 (acceptance): tiny-model smoke = SMOKE OK end-to-end. 1.5B vLLM
  pilot (20 greedy GSM8K rollouts): parse rate 1.000 at every position,
  prefix-cache hit 91.1% (>=90% gate PASS). Greedy argmax consistency 0.972
  (bf16: 0.966) -> gate amended pre-launch to >=0.95-with-margin-diagnostic
  (mismatch margins median 0.69 vs 3.7 at matches = benign cross-engine fp16
  numerics; rationale in README §6). forced_answer now shards rollouts
  (100/1.5B, 50/7B) so phase-A prefixes stay cache-resident, per plan. run_all
  script now takes config args (7B reuse), does the AIME 24k pre-pass, and
  widens forced_answer max_model_len to cover AIME traces.

- 2026-07-28 (GPU box): env up (venv /home/ubuntu/venvs/rc; pins resolved exactly
  to plan: torch 2.11.0 / transformers 5.14.1 / vllm 0.26.0). 47 tests green on
  GPU box. All 13 hub asset IDs verified incl. both AIME sets; weights prefetched
  to HF_HOME=/home/ubuntu/hf (1.5B/7B/Llama-8B/0.5B/32B-AWQ judge/SAE/L1).
  HF_TOKEN missing -> GPQA skipped by prepare_data (stages now skip absent
  manifests). Fixed before first real run: hendrycks_math config list,
  BatchEncoding token ids (hf_backend + phase_judge), manifest meta JSON,
  ActStore.gather positional scatter, forced_answer max_model_len + over-length
  prefix drop, MCQ answer regex anchoring, vllm cache-hit metrics, capture
  full-logits OOM (sliced argmax) + per-shard residual-norm accumulation.
- 2026-07-28: repo scaffolded, all modules + tests green on CPU; handoff.

## Progress log — 2026-07-28 19:55 UTC (D4 gate)
- Harness-tied background tasks were killed at 17:38; both pipelines relaunched **detached** (`nohup setsid`, logs at `runs/logs/run_{1p5b,7b}.log`) — survive session death; stages resumed from shard markers with no loss.
- **1.5B D4 gate: GO** — `GO/NO-GO: conv AUC=0.882 at L15; min within-stratum margin over shallow=0.147 => GO`. forced_answer complete on all 4 datasets (parse 1.000, ~95% cache hits). label_phase kappa: math_train 0.453, math500 0.472, gsm8k 0.524, aime 0.426 (moderate; report as labeling-noise caveat, phase results are secondary per plan). build_steering vectors orthogonalized (cos vs probe ≈ 0 after).
- 1.5B now in controller/baseline dev sweeps. 7B resumed mid-forced_answer (math_train done: 105k boundaries, parse 1.000, 94.8% hits); D4 gate for 7B pending.
