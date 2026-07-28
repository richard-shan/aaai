# ReasonControl: An Interpretable Latent-State Controller for Test-Time Compute
*Research plan + codebase design for AAAI. Adversarially reviewed by a 4-lens critic panel (novelty / feasibility / statistics / reviewer-2); all confirmed findings integrated below.*

## Context

Goal: a mechanistic-interpretability project on **optimizing reasoning language models** (R1-style long-CoT), targeted at AAAI. Facts established during planning:

- **Repo (`/home/user/repo`) is empty** — greenfield.
- **This planning container is CPU-only** (4 cores, 15 GB RAM); its network policy **blocks huggingface.co and arxiv.org** (PyPI allowed). Execution happens on a separate **2× H100 80GB** box (`~/aaai/aaai`). Codebase is written + unit-tested here (tiny-random models, vendored tokenizer fixtures); downloads, smoke-e2e, and all real runs happen on the GPU box.
- **Deliverable now:** rigorous design + complete runnable codebase + exact run commands; experiments over ~2 weeks.

## Direction & honest novelty positioning

**What we build:** a training-free controller that, at every reasoning-chunk boundary, reads the residual stream with **calibrated linear probes** — (a) *answer-convergence* (is the final answer already determined? labels via forced-answer truncation) and (b) *cognitive phase* (exploration/deduction/verification/backtracking) — and applies a state-conditioned action: **EXIT** (splice `</think>`, emit answer), **STEER** (causally-validated residual-stream direction), or **CONTINUE**.

**What we may NOT claim** (critic-verified prior work): closed-loop, state-conditioned intervention per se exists — **NEAT** (2602.02010: exit-neuron monitoring + reflection-token logit suppression), **AIC** (NeurIPS'25 wksp: XGBoost on embeddings, {continue, terminate, intervene}), **STU-PID** (2506.18831: redundancy classifier + PID-modulated steering), **DEER** (2504.15895: trial-answer confidence exit), **Dynasor/Certaindex** (2412.20993: forced-answer agreement exit, ships in vLLM), **TPV/Overclocking** (2506.07240: progress probe + same-direction steering), **CGRS** (2508.05337: certainty-gated reflection suppression at logit level). Must-cite adjacent: FlashThink 2505.13949, ES-CoT 2509.14004, answer-convergence stopping 2506.02536, SpecExit 2509.24248, future-behavior probes 2606.11172, control-point stability 2604.02113, Zhang probes 2504.05419, CREST 2512.24574, Venhoff taxonomy 2506.18167, Manifold Steering 2505.22411, L1/LCPO 2503.04697, ShorterBetter/O1-Pruner, TALE.

**What survives as the contribution (paper claims):**
1. **Interpretable, calibrated sensors with defined semantics** (convergence via forced-answer labels; phase via audited taxonomy) vs heuristic neuron patterns (NEAT), opaque classifiers (AIC), or extra-decode trial answers (DEER/Dynasor). Our probe costs **one matvec on states already computed** — zero extra forward passes; DEER/Dynasor pay a ≤32-token decode per check.
2. **Read–write duality with faithfulness evidence (★ core, not garnish):** the *same* directions are probed and causally steered; we show the convergence direction is not a position artifact (position-controlled AUC, shallow-feature baselines), that steering ±v_conv shifts propensity-to-conclude, and what the directions encode (logit-lens; SAE decomposition on Llama-8B via `qresearch/DeepSeek-R1-Distill-Llama-8B-SAE-l19`).
3. **A unified exit+steer policy** evaluated with symmetric protocol, paired statistics, and head-to-head Pareto against *implemented* members of every prior family — the first controlled comparison of latent-state probing vs trial-decode confidence vs static budgets vs static steering in one harness.
4. Phase channel has an action exit cannot do: when *not* converged and stuck in a sustained verification/backtracking loop (≥3 chunks), steer toward deduction to break the cycle (accuracy-relevant, not just token-saving).

**Hypotheses:**
- **H1:** conv+phase probes decode their targets **beyond position and shallow features** (AUC − position-baseline AUC ≥ 0.05 within position strata), calibrated (ECE < 0.1), incl. on uncontaminated anchors (AIME'25, GPQA).
- **H2:** phase directions AND v_conv are causal — validated with instruments independent of their construction (LLM-judge phase rates, behavioral endpoints, conclude-propensity), not the regex that built them.
- **H3:** the unified controller Pareto-dominates each single-channel family at matched protocol (primary: accuracy-at-matched-budget, tokens-at-matched-accuracy; pooled MATH+GSM8K endpoint, hierarchical test: token superiority → accuracy non-inferiority with pre-registered margin).
- **D6 framing gate (pre-registered):** if steering adds <5% of exit-only's savings at matched accuracy, re-headline as "latent-state monitoring for test-time compute control" with exit+steer as two instantiations; DEER comparison + faithfulness carry the contribution. Decided at D6, in writing — no post-hoc pivot.

### Models, data, assets
- **Models (★):** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (dev), `…-Qwen-7B` (main), `…-Llama-8B` (**promoted to ★**: probes + exit-only controller; SAE + full steering = fork). Optional generality row: Qwen3-8B thinking mode (probe AUC + exit-only).
- **Reference checkpoint (eval-only):** `l3lab/L1-Qwen-1.5B-Exact`/`-Max` — trained length-control specialist line on 1.5B Pareto plots.
- **Datasets & splits (fixed at `prepare_data`, seed=0; every test-time knob's dev set named here):**
  - MATH: probe-train (500) + **dev (200)** sampled from the MATH *training* split; **all of `HuggingFaceH4/MATH-500` untouched test**.
  - GSM8K (`openai/gsm8k` main): 300 probe-train / 50 dev / 250 test.
  - AIME: **'24 (n=30) = dev, '25 (n=30) = test** (post-dates R1 release → uncontaminated anchor). IDs: `AI-MO/aimo-validation-aime`, `math-ai/aime25` — *verify on GPU box; HF unreachable here*.
  - GPQA-Diamond (`Idavidrein/gpqa`, **gated — HF token needed**): OOD test-only; operating points inherited from MATH dev (pre-committed; no per-dataset test tuning).
- **Contamination policy:** MATH/GSM8K probe claims carry a caveat; H1 must hold on AIME'25 + GPQA. Diagnostic reported either way: rate of "converged by chunk 3" per dataset at matched difficulty (memorization signature).

---

## Engineering plan

### Repo structure

```
/home/user/repo
├── pyproject.toml            # reasoncontrol; PINNED: vllm==0.26.0, torch==2.11.0, transformers==5.9.0
├── Makefile                  # setup / test / smoke / smoke-gpu / stage shortcuts
├── README.md                 # GPU-box acceptance checklist
├── configs/
│   ├── base.yaml  datasets.yaml  smoke.yaml
│   ├── models/{r1_qwen_1p5b,r1_qwen_7b,r1_llama_8b,smoke_qwen_0p5b}.yaml
│   └── stages/…
├── src/reasoncontrol/
│   ├── config.py  paths.py
│   ├── data/{datasets.py, grading.py}          # math-verify equivalence everywhere; GPQA MCQ
│   ├── generation/{backend.py, vllm_backend.py, hf_backend.py}   # token-ids-only interface
│   ├── chunking.py                             # SHARED boundary detector (rolling 2-token tail)
│   ├── activations/{capture.py, store.py}      # safetensors + parquet; int8 flag
│   ├── labeling/{phase_regex.py, phase_judge.py, convergence.py}
│   ├── probes/{probe.py, train.py, calibration.py, shallow.py}   # shallow.py: position/cue-feature baselines
│   ├── steering/{vectors.py, hooks.py, validate.py}
│   ├── controller/{policy.py, loop.py, baselines.py}
│   ├── analysis/{pareto.py, stats.py, plots.py}
│   ├── interp/{logit_lens.py, sae.py, transfer.py}
│   └── stages/  (CLI entrypoints + _stage.py; idempotent, sharded by 50 problems, resumable)
├── scripts/{run_all_1p5b.sh, run_all_7b.sh, run_all_llama8b.sh, smoke_e2e.sh}
├── tests/                     # ALL CPU-runnable, no downloads (tiny-random Qwen2 + vendored toy-BPE tokenizer.json fixture with \n\n, .\n\n, \n\n\n merges)
└── runs/                      # gitignored
```

**Config:** plain YAML + frozen dataclasses (not Hydra); `load_config(paths, overrides)` with dotted `--set`; every stage dumps `resolved_config.yaml` + `stage.done` (git SHA, wall time). Sweeps = explicit YAML lists.

**Storage:** chunk-boundary residual states only, fixed layer set, fp16, `[n_boundaries, n_layers, d]` safetensors + parquet index. 1.5B ≈ 15 GB, 7B ≈ 34 GB, Llama-8B ≈ 39 GB. **GPU-box check: ≥250 GB free including HF_HOME (weights caches ~60 GB); point HF_HOME at the runs volume.**

### Core interfaces

As designed (Problem / RolloutRecord / GenBackend / ChunkRecord / LinearProbe / SteeringHook / PolicyConfig / ControllerPolicy / run_controlled_batch — see signatures below), with critic-driven amendments:

```python
class SteeringHook:              # transformers 5.x: layer forward may return bare Tensor
    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output   # handle both; mutate in place
        h[self._rows] += self._alphas[self._rows, None] * self._v_hat
    def set_rows(self, alphas): ...        # 0 = bit-exact no-op (skip path)

@dataclass(frozen=True)
class PolicyConfig:
    tau_exit: float; tau_steer: float; alpha: float
    patience_k: int; min_chunks: int
    steer_phases: tuple = ("verification", "backtracking")
    break_loops: bool = True     # NOT-converged + >=3 consecutive verify/backtrack chunks
                                 # -> steer toward deduction (phase channel's own action)
    max_think_tokens: int = 16384

# controller/loop.py — batched manual decode loop:
#  * StaticCache preallocated [batch, ~17.5k] (DynamicCache cat-per-step is quadratic; OOM risk)
#  * refill: prefill new rows in a separate small forward, slice-write KV into free slots at [0:len)
#  * explicit 2-D attention mask + per-row position_ids every step (no shared cache_position)
#  * finished rows never fully masked (attend to 1 token, discard) — SDPA NaN guard
#  * periodic compaction when >50% rows fresh; length-bucketed work queue
#  * per-row torch.Generator seeded by (problem_id, rollout_id, global_seed); multinomial per row
#  * EXIT answer segment decoded GREEDILY (matches the forced-answer labeling distribution)
#  * probe layer STRICTLY BELOW steering layer (sensor/actuator decoupling);
#    v_hat orthogonalized against conv- and phase-probe weight vectors (report cosines)
#  * hook on probe layer stashes last-position state; probe = one fp32 matvec at boundaries
```

**Reproducibility contract (amended):** NoOpPolicy vs same loop with hook detached and identical batch schedule = token-identical (unit-tested). NO claim of token identity vs `model.generate`, vs vLLM rollouts, or across conditions (batch-composition nondeterminism makes that unattainable); paired design needs only shared problems + per-row seeds, which this provides.

**Boundary detection:** ONE detector (rolling decoded 2-token tail) used by both offline `chunk_trace` and the online loop; GPU acceptance check: replaying stage-1 rollouts through the online detector reproduces `chunk_trace` boundaries exactly. Chunks: split thinking region on `\n\n`; merge <12 tokens into successor; cap 160 (keep first 10 + last 5). Boundary state = last token of chunk.

**Forced-answer probing (label engine, stage 4) — corrected:**
- Suffix: `"\n</think>\n\nThe final answer is \boxed{"` (math) / `"…is ("` (GPQA). **Never stop on `}`** (nested braces: `\boxed{\frac{3}{4}}`). Generate fixed ≤32 tokens (stop EOS/`\n\n` only); extract by **balanced-brace matching**; compare with **math-verify equivalence** (never string ==). CPU unit test with fraction/sqrt/interval/set fixtures.
- Labels: `conv_matches_final` (vs rollout's own final answer — primary, self-supervised) and `conv_correct` (vs gold — analysis + Zhang-style baseline).
- vLLM prefix caching is NOT automatic under chunked prefill co-scheduling → **two-phase submission per shard** (one LLM instance; cache persists across generate() calls): phase A = each rollout's longest-prefix probe only (disjoint), phase B = remaining ≤59 probes (hit block-aligned cached prefixes). Shard sizing: ~100 rollouts (1.5B) / ~50 (7B). **Pilot gate: prefix-cache hit rate ≥90% on 10 rollouts before full run.**
- `--audit` prints 50 samples; alternative suffixes are config strings.

**Steering vectors:** diff-of-means (`next chunk is verification` vs rest) on probe-train, unit-normalized, α in units of layer mean residual norm. **Stability filter (config flag, per 2604.02113):** for a subset of boundaries, resample 4 continuations; keep boundaries where the phase reproduces in a majority — fallback if raw-regex vectors fail validation.

### Execution DAG (amended stages/costs)

| # | Stage | Notes | H100-h (1.5B / 7B) |
|---|-------|-------|------|
| 0 | `prepare_data` | manifests, pinned revisions, split table above; asserts chat template auto-opens `<think>` | CPU |
| 1 | `generate` | vLLM, temp 0.6/top_p 0.95, 4 rollouts/problem, ≤16k think (+24k AIME) | ~1.5 / ~4 |
| 2 | `chunk` | shared detector | CPU |
| 3 | `capture` | teacher-forced HF pass, cached layers Qwen28: {9,12,15,18,21,24,26,27}, Llama32: {10,14,17,19,21,24,27,30}; consistency check: argmax reproduces ≥99% vLLM greedy tokens on 20 rollouts | ~0.5 / ~1.5 |
| 4 | `forced_answer` | two-phase prefix-cached; ≤60 boundaries/rollout | ~2 / ~9 |
| 5 | `label_phase` | regex all; LLM-judge (`Qwen/Qwen2.5-32B-Instruct-AWQ`) on 1.5k stratified chunks; κ reported | ~1 shared |
| 6 | `train_probes` | layer sweep; conv+phase probes; GroupKFold by problem; temp-scale on dev; **+ shallow.py baselines (position, chunk idx, last-chunk logprob/entropy, cue-token counts) and position-stratified AUC** | GPU-min |
| 7 | `build_steering` | diff-of-means + orthogonalization vs probe weights; **validation: behavioral effect over thousands of chunks w/ CI + LLM-judge phase-rate shift + paired accuracy non-inferiority on ≥200 paired dev rollouts (upper CI bound of drop <2%)**; + causal test of ±v_conv on conclude-propensity | ~2 / ~3 |
| 8 | `run_controller` | **grid on dev ONLY** (~16 configs × 200 MATH-dev + 50 GSM8K-dev); k operating points via **1-SE rule** on pooled dev; test runs at those points only; **8 seeds headline conditions, 4 for sweeps**; wall-clock/tokens-per-sec/GPU-s logged per condition; **closed-loop calibration audit**: stage-4 labeling on controller-generated dev trajectories at intervention boundaries → AUC/ECE under intervention gates the steering arm | ~12 / ~25 |
| 9 | `run_baselines` | see baseline suite; probe-free baselines (NoOp, StaticBudget, prompts) **run on vLLM** (~10× cheaper); probe/trial-decode baselines share the HF loop; same k-dev-selected-points protocol as controller | ~4 / ~10 |
| 10 | `analyze` | Pareto (below), CIs, ablations, wall-clock tables incl. NoOp-on-vLLM reference row + latency-to-answer distributions | CPU |
| 11 | `interp` (★ for 1.5B) | logit-lens on v_conv + phase directions; SAE decomposition (Llama-8B, qresearch l19); cross-model vocab-space transfer | ~2 |
| 12 | `vllm_exit_proto` (stretch) | minimal exit-only on vLLM: segment-wise generation (stop=`\n\n`, prefix caching, boundary probe via one teacher-forced forward); else a measured serving-overhead paragraph | ~2 |

Budget ≈ 30 (1.5B) + 55 (7B) + ~45 (Llama-8B ★ subset) + judge/pilots ≈ **~140 GPU-h vs ~670 available** — large headroom retained for reruns.

### Baseline suite (all `ControllerPolicy` subclasses in one harness; each family gets k dev-selected points)

1. **NoOp** (vLLM).
2. **StaticBudget(B)** = s1-style budget forcing: truncate at B, splice `\n</think>\n\n`, decode answer — never truncation-without-answer (vLLM).
3. **Prompting:** ConcisePrompt + **budget-conditioned prompts** B∈{1k,2k,4k} (TALE-style) (vLLM).
4. **ExitOnly-conv** (ours, probe on `\n\n` boundaries, patience k).
5. **ExitOnly-Zhang** (faithful variant: correctness target `conv_correct`, sentence-level chunking, their thresholding — claims "subsumes Zhang" only via this arm).
6. **Trial-decode exit (DEER/Dynasor-style):** at boundaries, force answer suffix; exit on k consecutive agreeing forced answers / high answer confidence. Reuses stage-4 machinery; report its per-boundary compute cost (≤32-token decode) vs our matvec — **this head-to-head is a central result.**
7. **SteerOnly:** α sweep of our vector; + **TPV-style arm** (progress-direction steering, same hook, different label source).
8. **L1-Qwen-1.5B** checkpoints, eval-only reference line (trained specialist).
9. *Nice-to-have:* NEAT-style (reflection-token logit suppression + exit), CGRS arm (logit suppression gated on our conv probe — isolates "where to intervene": logits vs residual stream).

### Statistical plan (amended)

- Unit = problem; identical problems + per-(problem,rollout) seeds across conditions; cluster bootstrap (10k) for all CIs.
- **Primary endpoint (pre-registered):** pooled MATH-500 + GSM8K test (n≈750), stratified cluster bootstrap, **hierarchical**: (1) token superiority, then (2) accuracy non-inferiority. **Margin pre-registered at D6** after measuring the actual paired SE on dev (plan: 8 seeds on headline conditions → SE ≈ 0.6%, power ~0.85–0.9 at 2%).
- **Pareto methodology:** symmetric k-points-per-family protocol; interpolation only on the upper-left convex hull, justified by randomized policy mixing (mixture weights applied per problem inside each bootstrap resample → paired CIs at a budget are well-defined); hypervolume demoted to secondary (min-max normalized axes, reference = (max tokens, min acc − ε), hull recomputed per resample); **primary metrics: accuracy-at-matched-budget {2k,4k,8k} and tokens-at-matched-accuracy.**
- AIME'25 n=30 / GPQA n=198: **descriptive paired deltas with CIs only** — no non-inferiority claims (unpowered); formal OOD claims = probe AUC transfer + token reduction. "Hard" analysis = AIME + MATH-L5 pooled.
- Probe reporting: AUC/ECE with clustered CIs, **minus position-baseline within strata**; chunks-early curves; phase macro-F1 vs regex and judge; κ reported.
- **Go/no-go (D4): conv probe beats position baseline by ≥0.05 AUC within-stratum AND AUC ≥ 0.75 on MATH dev.**

### Risk register (amended)

| Risk | Trigger | Fallback |
|---|---|---|
| Conv probe = position artifact | fails position-controlled go/no-go | project remains: trial-decode exit comparison + phase steering + faithfulness negative result is still a paper ("what probes actually decode"); pivot decided at D4 |
| Steering hurts accuracy / fails closed-loop audit | paired CI gate or audit fails | gate on p(conv)≥τ_steer; halve α; stability-filtered vectors; D6 framing gate re-headlines exit-led |
| Steering adds <5% over exit-only | D6 gate | pre-registered re-framing (above) |
| Probe AUC collapses on AIME'25 | <0.7 | report as contamination finding (itself informative); hard-pool analysis |
| vLLM prefix-cache hit rate <90% | pilot | shrink shards; sort within rollout; worst case eat 2-9 GPU-h recompute |
| transformers 5.x API drift | CI on GPU box | pinned trio (vllm 0.26.0/torch 2.11.0/transformers 5.9.0); hook handles Tensor|tuple; tests written against 5.9.0 in THIS container |
| Controller loop slow | >2 d projected | StaticCache+compile already designed; dev-only sweeps; 7B at 3 points; probe-free baselines already on vLLM |
| GPQA gated access fails | download error | OlympiadBench subset |
| Regex phase labels noisy | κ<0.5 | judge-labeled subset (1.5k→5k chunks) trains phase probe |
| Forced-answer garbage at early boundaries | audit | alternative suffixes (config); report parse-rate by position |
| Truncation inflates wins | >10% | 24k AIME cap; truncation rates in every table |
| Disk | >150 GB | 4-layer post-sweep; int8 flag; ≥250 GB acceptance check |

### Verification

- **CPU unit tests** (`pytest -m "not gpu"`, <2 min, zero downloads): chunker round-trip + token alignment on **vendored toy-BPE with `\n\n`/`.\n\n`/`\n\n\n` merges**; balanced-brace forced-answer extraction fixtures; math-verify grading fixtures; probe planted-direction recovery + calibration; hook row-masking exactness + `set_rows(0)` bit-exact no-op + Tensor|tuple handling; policy state machine (patience, min_chunks, hysteresis, EXIT-once, break_loops); controller loop on tiny-random Qwen2 (StaticCache refill, finished-row guard, NoOp≡hook-detached token identity); ActStore round-trip; probe-matvec <1 ms.
- **`make smoke`** (stages 0–10, Qwen2.5-0.5B, 5 problems, `think_tags=off`) — requires HF: runs as **first command on GPU box** (`make smoke-gpu` with real 1.5B + vLLM follows). This container: `make test` only.
- **GPU-box acceptance checklist (README):** disk ≥250 GB incl. HF_HOME → asset-ID verification (AIME/GPQA/L1/SAE repos) → smoke-gpu → vLLM/HF consistency ≥99% → forced-answer audit (parse rate by position) → prefix-cache pilot ≥90% → online-vs-offline boundary equivalence → 50-problem probe sanity (AUC>0.6) → launch.

### Schedule (2 weeks; ★ = minimum publishable subset)

- **D1–2 (this session):** scaffold + all modules + tests green (transformers 5.9.0 installed here); ship to GPU box.
- **D3:** acceptance checklist; launch 1.5B stages 1–4 all datasets; 7B generate on GPU 2.
- **D4:** 1.5B labels + ★probes + shallow/position baselines → **go/no-go**; 7B capture/forced in background; Llama-8B generate starts.
- **D5:** ★1.5B steering + causal validation (incl. v_conv test); 7B labels/probes.
- **D6:** ★1.5B dev sweeps (controller + all baseline families) + closed-loop audit → operating points (1-SE); **framing gate + margin pre-registration, in writing**; trial-decode + Zhang + TPV arms run here.
- **D7:** ★1.5B test runs (8-seed headliners); first Paretos + wall-clock tables; buffer.
- **D8:** 7B steering + dev sweep; Llama-8B probes.
- **D9:** ★7B test runs; ★Llama-8B exit-only + probe transfer.
- **D10:** ★OOD (GPQA, AIME'25 anchors) + ablations (exit/steer/full, layer, probe arch, patience, chunk granularity); L1 reference runs.
- **D11:** ★interp: logit-lens on v_conv (1.5B), SAE (Llama-8B), cross-model transfer; *stretch:* vLLM exit prototype; NEAT/CGRS arms.
- **D12:** statistics pass; contamination diagnostics; hypervolume secondary.
- **D13–14:** figures/tables freeze, writing, reserve rerun slot.

**★ Core claims:** H1 with position control on 1.5B+7B+Llama-8B (probe AUC also on AIME'25/GPQA anchors); H2 with judge-validated causality incl. v_conv; H3 pooled primary endpoint with symmetric Pareto protocol vs {NoOp, budget forcing, budget prompts, ExitOnly-conv, ExitOnly-Zhang, trial-decode exit, SteerOnly(+TPV), L1 reference}; component ablation; faithfulness section (logit-lens + shallow-feature comparison); wall-clock honesty table. **Nice-to-have:** SAE features, Qwen3-8B row, NEAT/CGRS arms, vLLM prototype.

### Critical files
- `src/reasoncontrol/controller/loop.py` — StaticCache batched decode loop (highest-risk code)
- `src/reasoncontrol/chunking.py` — shared online/offline boundary detector
- `src/reasoncontrol/labeling/convergence.py` — forced-answer engine (labels + trial-decode baseline)
- `src/reasoncontrol/steering/hooks.py` — per-row masked steering, 5.x-safe
- `src/reasoncontrol/probes/shallow.py` — position/shallow baselines (H1's defense)
- `src/reasoncontrol/config.py` — config layer
