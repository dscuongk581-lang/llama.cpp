# VLANIX P3.4 — Static Row Ownership vs GGML Atomic Work Queue on VPS02

Date: 2026-11-13. Isolated lab branch; **no merge, production deployment, model modification or service restart**.

## Hypothesis

After P3.3 introduced an adaptive 1/2/4/8 logical Micro-VCPU row-shard scheduler, atomic queue claims and worker synchronization were suspected sources of latency. P3.4 tests a competing strategy: **assign contiguous row-shard IDs to an existing worker** and avoid atomic task claims. This tests *worker ownership / locality*, not distributed computing across devices.

## Implementation

Within the existing GGML CPU `GGML_OP_MUL_MAT` path, for the already-eligible Q4_0×Q8_0 SSE2 matvec and under compile-time `GGML_VCPU_LAB=ON`, a new **opt-in-only** `GGML_VCPU_MICRO_SCHED=static` flag activates contiguous ownership. The worker index owns a block of `ceil(number_of_logical_tasks / number_of_workers)` row shards, and advances through them without the shared GGML `current_chunk` atomic fetch-add.

Micro-VCPU mode itself still requires `GGML_VCPU_LAB_ENABLE=1 GGML_VCPU_MICRO_ENABLE=1 GGML_VCPU_MICRO_TASKS=adaptive|8`. With all flags unset the unmodified native path is used. With the static flag unset P3.3's default adaptive atomic scheduler remains in effect. No additional OS threads or model weight copies are created.

Initial test was deliberately **RED** against the P3.3 binary (no static scheduler trace); after implementation/build P3.4's real graph tests were **GREEN** for 1/2/4/8 logical tasks and multivector fallback. Original hook and fixed8 tests continue to work. CI asserts the static trace, graph parity and fallback, and in instrumented static mode requires **zero atomic task claims**.

## Reproducible real Qwen decode comparison

Same fork-compiled `llama-bench`, same `Qwen3.5-0.8B-Q4_0.gguf` stored on VPS02, same two CPU workers, SSE2-only baseline build. Compare FIVE configurations, five seeded randomized repetitions (25 observations); 24 generated tokens and two internal `llama-bench` repetitions per observation. Instrumentation disabled for speed measurement; one-shot trace verifies actual routes.

| Mode | 2-worker median generated tok/s |
|---|---:|
| Native GGML CPU | **3.0540** |
| SSE2 hook without extra task partition | **4.4921** |
| Adaptive P3.3 with GGML atomic task queue | **4.1731** |
| Adaptive row task sizes with **static worker ownership** | **3.7366** |
| Fixed-8 logical row tasks with static ownership | **4.1063** |

**Paired speed ratios (Adaptive Static / control):**

| Baseline | Median ratio | Static wins |
|---|---:|---:|
| Native 2-worker | 1.2415× | 4/5 |
| SSE2 direct hook 2-worker | **0.8277×** | **0/5** |
| Adaptive Atomic 2-worker | **0.8833×** | **0/5** |
| Fixed8 Static 2-worker | 0.8925× | 1/5 |

**Finding: eliminating atomic queue claims makes Qwen slower in all five matched comparisons against the adaptive atomic scheduler and the simpler direct SSE2 hook.** This is negative evidence for static ownership in the tested two-vCPU conditions, not a proof that dynamic work stealing always wins. The high variability in a shared VPS, five observations, short decode and single CPU ISA preclude generalizations.

Raw CSV: `docs/lab-results/vps02-p34-static-qwen-decode-20261113.csv`. Validator `scripts/vcpu-p34-validate.py` (three unit tests PASS VPS02) and aggregate evidence: `docs/lab-results/vps02-p34-static-qwen-analysis-20261113.json`.

## True Qwen numerical equivalence

The P3.2 direct `llama_decode` comparison binary was run with P3.4's `GGML_VCPU_MICRO_SCHED=static` enabled. Two fresh sequential Qwen contexts (native and static), one synthetic arithmetic prompt, 16 greedy generated steps and full 248,320-vocabulary logits per step:
- One worker: 16/16 greedy token IDs match native, printed maximum absolute logits discrepancy `0.000000000`, no invalid mismatches, PASS.
- Two workers: 16/16 greedy token IDs match native, same results, PASS.

Source: `docs/lab-results/vps02-p34-static-qwen-logit-parity-20261113.json`. A short greedy example does not establish universal numerical parity, stochastic-sampling equivalence, long-context correctness or production readiness.

## Synchronization instrumentation, separated from performance runs

Two separate **single-run instrumented** Qwen decodes, eight generated tokens, same two GGML workers. Their timing diagnostics are **not** unbiased benchmarks.

| Diagnostic | Static Adaptive | P3.3 Atomic Adaptive |
|---|---:|---:|
| Real logical task executions | 7,776 | 7,776 |
| Atomic queue claims | **0** | **7,560** |
| Total atomic claim measured | **0 ns** | 1,468,586 ns |
| Total worker compute duration, summed | 2,905,248,945 ns | 2,238,654,542 ns |
| Mean elapsed GGML barrier time / worker visit | **6,864.55 ns** | **2,956.59 ns** |
| Actual adaptive Qwen task mix | 108×2, 594×4, 648×8 | same |

Static mode achieves its mechanical objective: 0 atomic queue claims. But static ownership's sum of measured task computation and barrier elapsed time was higher in this *single noisy instrumented comparison*. **We cannot assign causation from these two runs:** wall-time and worker time differ, compute times overlap across workers, barrier elapsed includes arrival skew, and instrumentation itself adds overhead. Their direction is consistent with the 25 speed measurements, and motivates work-balance investigation.

Diagnostics: `docs/lab-results/vps02-p34-static-vs-atomic-sync-20261113.json`.

## CI, controls and next gate

The fork CI workflow `.github/workflows/vlanix-vcpu-lab.yml` builds GGML with `GGML_VCPU_LAB=ON` and `OFF`, runs baseline operator tests, P3.2 fixed8, P3.3 adaptive and P3.4 static 1/2/4/8/fallback graph tests, and verifies zero queue claims under static instrumented mode. All model-size experiments remain on VPS02 and do **not** commit GGUF data.

**Decision: NO PROMOTION.** Keep **native default** and the hook/micro/static experimental flags OFF unless explicitly selected in lab. On this host the direct SSE2 hook was faster than either Micro-VCPU scheduler. Do not optimize for queue claim count alone.

Recommended P3.5: instrument per-worker row-shard execution distribution and imbalance, consider a hybrid locality-preserving bounded work-stealing scheduler, and only allow task splitting for shapes where a controlled holdout proves speedup against **direct SSE2**, not merely against native. When local dynamic scheduling is validated, revisit cross-node scheduling and transfer costs.
