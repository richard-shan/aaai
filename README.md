# ReasonControl

**An interpretable latent-state controller for test-time compute in reasoning
LLMs.** At every reasoning-chunk boundary, calibrated linear probes read the
residual stream — (a) *answer convergence* (is the final answer already
determined?) and (b) *cognitive phase* (exploration / deduction / verification
/ backtracking) — and a policy applies a state-conditioned action: **EXIT**
(splice `</think>`, decode the answer greedily), **STEER** (causally-validated
residual-stream direction), or **CONTINUE**. Probe cost: one matvec on states
the forward already computed — zero extra model passes.

Full research plan (hypotheses, baselines, statistics, risk register,
schedule): see the plan document accompanying this repo.

## Layout

```
src/reasoncontrol/
  chunking.py        shared online/offline boundary detector + chunker
  data/              manifests, prompts, math-verify grading
  generation/        token-ids-only GenBackend (vllm | hf)
  activations/       teacher-forced boundary capture; safetensors ActStore
  labeling/          forced-answer convergence labels; phase regex + LLM judge
  probes/            linear/MLP probes, calibration, shallow/position baselines
  steering/          diff-of-means vectors, orthogonalization, per-row hooks
  controller/        policy state machine, batched HF decode loop, vLLM baselines
  analysis/          cluster bootstrap, Pareto/hypervolume
  interp/            logit lens, SAE decomposition (Llama-8B l19), transfer
  stages/            CLI entrypoints (the execution DAG below)
configs/             base.yaml + model/smoke overlays
scripts/             run_all_1p5b.sh, smoke_e2e.sh
tests/               CPU-only unit tests (no downloads)
```

## Setup

```bash
make setup          # CPU dev (tests only)
make setup-gpu      # GPU box: adds vllm + datasets
make test           # 47 unit tests, <30 s, no network
```

Pins: `transformers==5.14.1` (code is written against 5.x APIs: bare-Tensor
layer outputs, StaticCache(config, max_cache_len)). On the GPU box let vllm
drive the torch pin; if pip resolves a transformers other than 5.14.x, re-run
`make test` there before anything else.

## GPU-box acceptance checklist (run in order, first session)

1. `df -h` — need **>= 250 GB free** on the volume holding `runs/` AND
   `HF_HOME` (point `HF_HOME` at the same volume; weights ≈ 60 GB).
2. `export HF_TOKEN=...` with accepted terms for `Idavidrein/gpqa` (gated).
3. Verify asset ids resolve: the four `deepseek-ai/DeepSeek-R1-Distill-*`
   models, `HuggingFaceH4/MATH-500`, `openai/gsm8k`,
   `AI-MO/aimo-validation-aime`, `math-ai/aime25`,
   `qresearch/DeepSeek-R1-Distill-Llama-8B-SAE-l19`, `l3lab/L1-Qwen-1.5B-Exact`.
4. `make test` then `make smoke` (tiny model, end-to-end DAG).
5. `make smoke-gpu` (real 1.5B, 1 rollout, 2k tokens) — checks the chat
   template auto-opens `<think>`, vLLM loads, and generation round-trips.
6. Stage `capture` must report **argmax consistency >= 0.99** (vLLM/HF
   numerics check); stage `forced_answer --audit` parse rates should be sane
   at early boundaries; the vLLM prefix-cache hit rate on a 10-rollout pilot
   must be **>= 90%**.
7. Launch `bash scripts/run_all_1p5b.sh` under `CUDA_VISIBLE_DEVICES=0` and
   the 7B equivalent (`--config configs/base.yaml configs/models/r1_qwen_7b.yaml`)
   under `CUDA_VISIBLE_DEVICES=1`.

## Execution DAG

| stage | reads → writes | notes |
|---|---|---|
| prepare_data | HF hub → `runs/data/manifests/*.parquet` | fixed splits, seed 0; AIME'24=dev, AIME'25=test |
| generate | manifests → `rollouts/` | vLLM, 4 rollouts/problem, temp 0.6 |
| chunk | rollouts → `chunks.parquet` | shared boundary detector |
| capture | rollouts+chunks → `acts/` | boundary states only, 8 layers, fp16 |
| forced_answer | rollouts+chunks → `forced.parquet` | two-phase prefix-cached; balanced-brace parsing |
| label_phase | chunks → `labels.parquet` | regex all + judge subset (kappa) |
| train_probes | acts+labels → `probes/` | layer sweep + shallow/position control (**go/no-go**) |
| build_steering | acts+labels → `steering/` | diff-of-means, orthogonalized vs probe dirs |
| run_controller | probes+steering → `controller/` | HF loop; `RC_SPLIT=dev|test`; grid on dev ONLY |
| run_baselines | manifests → `controller/` | vLLM: noop / budgets / prompts / trial-decode |
| analyze | controller/ → `runs/analysis/` | Pareto, paired bootstrap, non-inferiority |
| interp | probes+steering → `analysis/interp.json` | logit lens; SAE on Llama-8B |

Every stage: `python -m reasoncontrol.stages.<name> --config configs/base.yaml
[configs/models/X.yaml] [--set key.sub=val] [--force] [--datasets ds ...]`.
Stages are idempotent and resume at shard granularity.

## Protocol guardrails (do not bend these)

- Sweeps run on **dev only** (`RC_SPLIT=dev`); test runs happen once, at
  dev-selected operating points (1-SE rule), 8 seeds for headline conditions.
- The go/no-go for the convergence probe is **position-controlled**: pooled
  AUC >= 0.75 AND >= 0.05 AUC over the shallow baseline within every position
  stratum. A probe that only re-encodes position does not pass.
- AIME'25 and GPQA are the uncontaminated anchors: probe claims must
  replicate there; their accuracy comparisons are descriptive CIs only.
- Steering acceptance is the paired criterion in `steering/validate.py`
  (upper CI bound of accuracy drop < 2% on >= 200 paired dev rollouts), plus
  judge-verified phase-rate shift — never the regex alone.
