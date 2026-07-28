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
