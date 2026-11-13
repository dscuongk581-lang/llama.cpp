# VLANIX P3.5 — Locality-First Bounded Queue Scheduling in Qwen / GGML

Date: 2026-11-13. VPS02 (2 QEMU vCPUs, SSE2-only baseline), isolated fork lab. **Draft-only, not production or cross-node distribution.**

## Research hypothesis

P3.3 adaptive scheduler used the GGML CPU shared atomic queue for additional Micro-VCPU row tasks. P3.4 assigned all shards statically and eliminated shared claims, but **lost** throughput. Test a compromise: reserve small contiguous, cache-friendly work ranges for each existing GGML worker, then use the shared queue for the remaining work to preserve some dynamic balancing.

This is **locality-first preclaim + shared work queue**, NOT a complete independently addressable work-stealing system and NOT distributed VCPU across VPSs.

## Actual code change

In the real GGML CPU `GGML_OP_MUL_MAT` pathway, opt-in `GGML_VCPU_MICRO_SCHED=hybrid` when the existing P3.0 hook + P3.3 Micro-VCPU gates are active.

- At most two adjacent logical row-shard tasks are reserved per physical worker; for 2-task shape, one per worker; for 4-task shape, two per worker; for 8-task shape, two per worker and four remain in the shared atomic queue. With only one worker, the single task is local.
- Assign worker `ith` an exclusive initial range `[ith * local_count, (ith + 1) * local_count)`.
- Set GGML queue start to `worker_count * local_count` before GGML's existing barrier; no duplicate or missing task by construction.
- Upon exhausting the reserved initial range, claim the remaining work with GGML's existing atomic queue, not a new nested worker pool.
- Compile-time `GGML_VCPU_LAB=OFF` and runtime flags OFF by default; all unsupported shapes / batches / CPU modes remain on GGML native. `adaptive`, `static` and fixed `8` remain explicit lab controls.
- This is deliberately not a network transport, a GGML backend device, or an OptiBlockAI buffer system.

## Test-driven correctness

An expected-RED test on the P3.4 binary confirmed absence of the hybrid trace; after P3.5 code was added to the isolated fork, compile passed and the real GGML graph tests passed all 5 shape/worker cases: 512 rows→2 tasks/2 workers, 1024→4/2, 3584→8/2, 3584→1/1, multi-vector fallback without special scheduling. All outputs matched GGML native within numeric test tolerance.

`llama_decode` real Qwen3.5-0.8B-Q4_0 tests, one synthetic short prompt each with 16 greedy steps:
- **Hybrid 1 worker:** all 16 token IDs equal, 248,320-vocabulary logits maximum printed absolute difference `0.000000000`, invalid mismatch count zero, PASS.
- **Hybrid 2 workers:** same, PASS.
- The precision is nine decimal places and the test covers only one short prompt per worker mode; it is not broad model equivalence.
- Evidence: `docs/lab-results/vps02-p35-hybrid-qwen-logits-20261113.json`.

## Whole-Qwen inference benchmark

5 randomly ordered repeated blocks × 5 routes = **25 observations**, same fork-built `llama-bench` linked to the P3.5 GGML CPU lab. Real local Qwen3.5-0.8B Q4_0 GGUF, 2 GGML worker threads, 24 generated tokens, 2 internal benchmark repetitions, seed 350113. Instrumentation disabled for these timing runs.

| Route | Median generated tok/s |
|---|---:|
| GGML Native (2 workers) | **3.1203** |
| SSE2 direct hook (2 workers, no added shards) | **4.3386** |
| P3.3 Adaptive Atomic (2 workers) | **4.3861** |
| P3.4 Adaptive Static (2 workers) | **4.0222** |
| P3.5 Adaptive Hybrid (2 workers) | **4.0938** |

Paired ratios **Hybrid / control**:
- Native 2 workers: median **1.3579×**, 4/5 wins.
- Direct SSE2 hook: median **0.9382×**, 1/5 wins.
- Adaptive Atomic: median **0.9518×**, 2/5 wins.
- Adaptive Static: median **0.9691×**, 2/5 wins.

**Conclusion:** Hybrid does not improve performance over simpler SSE2 hook or P3.3 Adaptive Atomic on this host in this sample. Do not promote it into the default kernel path. Five paired repetitions on a noisy shared VPS are exploratory, not conclusive for other models, CPU ISAs or long-context behavior.

Raw data: `docs/lab-results/vps02-p35-hybrid-qwen-decode-20261113.csv`.
Strict CSV/trace/paired-sample validator `scripts/vcpu-p35-validate.py` (3 unit tests PASS on VPS02).
Summary: `docs/lab-results/vps02-p35-hybrid-qwen-analysis-20261113.json`.

## Actual scheduling overhead, instrumented separately

Three **separate, single-run diagnostic** Qwen decodes, 8 generated tokens, 2 worker threads, `GGML_VCPU_MICRO_STATS=1`. Same 1,350 eligible operator invocations and **7,776 logical tasks** per mode (108 × 2 tasks, 594 × 4 tasks, 648 × 8 tasks).

| Diagnostic | Adaptive Atomic | P3.5 Hybrid | Adaptive Static |
|---|---:|---:|---:|
| Atomic queue claims | **7,560** | **5,076** | **0** |
| Aggregate queue claim elapsed | 1,286,745 ns | 924,621 ns | 0 ns |
| Mean GGML barrier elapsed/worker arrival | 2,758.73 ns | 3,927.58 ns | 3,275.80 ns |

Hybrid reduces atomic claims by **32.86%** vs Adaptive Atomic as designed. Nevertheless, the short whole-model speed benchmark remains slower; less queue traffic alone is not the governing objective.

**Timing caveats:** Instrumented runs add overhead; worker compute durations overlap, barrier duration includes worker-arrival skew, host load varies between sequential runs, and aggregate durations cannot be added directly to a wall-time speedup. These are diagnostic counters, not unbiased attribution or proof of causation.

Full diagnostic JSON: `docs/lab-results/vps02-p35-hybrid-vs-atomic-sync-20261113.json`.

## Acceptance and next step

P3.5 passes research-only integration correctness, model-level short greedy token/logit parity, and demonstrates real reduction in queue claims. **It fails the incremental whole-model throughput test against direct SSE2 hook and adaptive atomic** in this sample.

Keep PR Draft, keep all opt-in flags OFF in regular builds, no merged changes or production deployment. Do not label this as full work stealing or distributed acceleration.

**P3.6 recommendation:** measure per-worker execution progress, row-shard completion imbalance and cost per tensor shape. Evaluate a limited cost-aware policy that selects the *plain SSE2 kernel without Micro-VCPU splitting* by default, and only selects another scheduling path when a sufficiently broad independent heldout experiment shows a meaningful speedup. Only then add inter-node latency as a new variable.
