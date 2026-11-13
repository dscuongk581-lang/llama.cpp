# VLANIX P3.3 — Adaptive Local Micro-VCPU Scheduler (VPS02)

Date: 2026-11-13. Single VPS02 lab, QEMU 2 vCPU with SSE2 baseline, Qwen3.5-0.8B Q4_0. **Draft research, not distributed inference, not production.**

## Objective and change

P3.2 proved eight logical row shards can be scheduled by the real GGML CPU graph and 1/2 physical workers but yielded **no reliable incremental gain** compared with the direct SSE2 CPU hook.

P3.3 introduces **opt-in adaptive chunk count** for eligible Q4_0 × Q8_0 GGML `MUL_MAT` operations. Unlike eight fixed chunks it selects based on number of output rows and worker count:

| GGML physical workers | Q4 output rows | Logical Micro-VCPU row tasks |
|---|---|---|
| 1 | eligible | **1** (no extra atomic queue claims) |
| 2 | 128–1,023 | **2** |
| 2 | 1,024–3,071 | **4** |
| 2 | ≥3,072 | **8** |

This is a **predeclared heuristic, not a trained runtime autotuner**. Each logical task operates on a contiguous row shard. It reuses GGML's native threadpool/atomic work queue and existing P3.0 SSE2 Q4×Q8 dot kernel; **no nested thread pool or 8 physical threads**.

Gate: `GGML_VCPU_LAB=ON` when building and all three runtime envs `GGML_VCPU_LAB_ENABLE=1`, `GGML_VCPU_MICRO_ENABLE=1`, `GGML_VCPU_MICRO_TASKS=adaptive`. Any unsupported tensor/worker count, additional vector, geometry, ISA or disabled setting follows GGML's original path. P3.2 `GGML_VCPU_MICRO_TASKS=8` is retained as a fixed control. Default build and runtime settings remain OFF.

## TDD and correctness

- RED first: the P3.2 lab binary did not emit adaptive dispatch trace.
- GREEN after isolated fork build: `tests/test-vcpu-p33-adaptive.cpp` PASS for 512 rows→2 tasks/2 workers, 1,024→4 tasks/2 workers, 3,584→8 tasks/2 workers, large tensor→1 task/1 worker and multi-vector fallback with no adaptive task dispatch (**5/5**).
- Native vs custom graph F32 output max abs error remains below 0.002.
- Real Qwen model via direct `llama_decode`, one short arithmetic prompt, 16 greedy steps, with **full 248,320-vocab F32 logits** checked:
  - Adaptive 1 worker: **16/16 token IDs match**, printed max and mean abs logits difference `0.000000000`, zero invalid mismatches.
  - Adaptive 2 workers: **16/16 token IDs match**, same result.
  - These are single-short-prompt checks with 9-decimal printed precision, not universal equivalence proof.
- Result evidence: `docs/lab-results/vps02-p33-adaptive-logits-parity-20261113.json`.
- GitHub CI compiles feature ON/OFF, checks adaptive task selection/fallback, and runs validator unit tests. No private GGUF bytes uploaded.

## Qwen decode performance

Same fork-built `llama-bench` and same isolated dynamically linked GGML CPU library (built from the P3.3 lab fork before additional **opt-in-only diagnostic** counters were committed). 5 random-order repeats × 5 profiles = **25 real trials**. Each trial: Qwen3.5-0.8B Q4_0, all CPU, 24 generated tokens, two internal repetitions, 1 or 2 GGML worker threads. Seed 330113. Measurements intentionally have **stats instrumentation disabled**, and exact-mode hook/dispatch traces are checked.

| Profile | Workers | Median tok/s | Range |
|---|---:|---:|---|
| GGML native | 2 | **3.3495** | see CSV |
| SSE2 hook without Micro-VCPU chunking | 2 | **4.4649** | see CSV |
| P3.2 fixed 8 Micro-VCPUs | 2 | **4.0952** | see CSV |
| P3.3 adaptive task count | 2 | **4.4379** | see CSV |
| P3.3 adaptive task count | 1 | **2.9786** | see CSV |

**Paired median throughput ratios (adaptive 2 worker / control):**
- vs native 2 worker: **1.4029×**, 5/5 wins.
- vs P3.2 fixed8 2 worker: **1.0556×**, 3/5 wins.
- vs direct SSE2 hook 2 worker: **0.9699×**, 2/5 wins.

This is one shared noisy VPS; 5 repeats are exploratory and insufficient for strong performance claims. **Adaptive improved over fixed-8 in the observed median but did not reliably beat the simpler SSE2 hook.** No reason to enable in production yet.

Raw observations: `docs/lab-results/vps02-p33-adaptive-qwen-decode-20261113.csv`; strict validator: `scripts/vcpu-p33-validate.py` (3 tests PASS); summarized ratios: `docs/lab-results/vps02-p33-adaptive-qwen-analysis-20261113.json`.

## Real adaptive task mix and coordination diagnostics (separate run)

P3.3 added optional diagnostic counters `GGML_VCPU_MICRO_STATS=1`; **these were not active during speed tests**. On a separate 8-token Qwen decode and two workers:

| Observed number of logical tasks | Eligible operator executions |
|---|---:|
| 1 | 0 |
| 2 | **108** |
| 4 | **594** |
| 8 | **648** |
| Total | **1,350** |

The model actually executed **7,776 logical row tasks** rather than a fixed 10,800 (a reduction of 28.0% in logical tasks), confirming that adaptive selection is exercised during real Qwen decode.

Measurements in this *instrumented single run*:
- Total atomic queue claims: **7,560**, mean timing **234.20 ns/claim**.
- GGML barrier mean elapsed **7,396.48 ns per worker visit**, includes arrival skew; this is **not exclusively intrinsic synchronization overhead**.
- Worker compute times overlap, so their sum cannot be added to the program's wall time. The stats-enabled run is **not** a valid replacement for token/s measurements.
- Evidence: `docs/lab-results/vps02-p33-adaptive-qwen-task-distribution-20261113.json`.

## Design conclusion and next step

P3.3 proves a dynamic, size-aware Micro-VCPU row-shard scheduler can operate on the **actual Qwen GGML graph** with numerical parity over the tested cases. The extra scheduler improves the fixed8 choice in the observed sample, but the default choice should remain **GGML native with the experimental SSE2 hook available only via explicit lab opt-in**; adaptive is not yet a proven improvement over the direct SSE2 hook.

**P3.4 next:** compare static worker affinity, row-shard cache locality and eliminate queue claims when task count equals workers; measure this against direct SSE2 at *the same two-worker count* across broader prompt/context lengths, with enough heldout runs to choose automatically only after a genuine speed margin. Do not infer any cross-node distributed Qwen speedup from these single-host observations.

All changed files and evidence remain on the Draft PR; no model file contents, system libraries, Qwen production executable, server configuration or resident services were modified.
