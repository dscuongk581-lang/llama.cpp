# VLANIX P3.2 — Local Multi-VCPU (8 logical tasks / 1 or 2 GGML workers)

Date: 2026-11-13, VPS02 isolated lab; **Draft-only**, no production deployment.

## Objective and architecture

Test whether splitting a Qwen matvec operation into **8 logical Micro-VCPU tasks**, executed by the existing **1 or 2 GGML CPU worker threads**, improves inference over native and over the P3.0 SSE2 hook without extra task partitioning. This is single-host scheduling, **not** distributed tensor computation across multiple devices.

Implementation `ggml/src/ggml-cpu/ggml-cpu.c`:
- In the real `GGML_OP_MUL_MAT` CPU graph execution, after conversion to GGML Q8_0 activations, when strict eligibility and opt-in gates pass, set `nchunk0=8`, `nchunk1=1`. This produces eight *contiguous row-shard* logical tasks and uses GGML's already-existing atomic chunk claim queue. First 1/2 workers start at their own initial tasks, subsequent tasks are claimed through the same queue.
- No extra OS threads, no nested threadpool, no model-format changes. Existing P3.0 quantized SSE2 `vec_dot` hook remains the numerical kernel.
- All gates must hold: compile-time `GGML_VCPU_LAB=ON`, runtime `GGML_VCPU_LAB_ENABLE=1`, `GGML_VCPU_MICRO_ENABLE=1`, `GGML_VCPU_MICRO_TASKS=8`, contiguous Q4_0 weights and F32 activation with Q8_0 dot type, 1-vector mode, at least 128 rows, columns multiple of 32, 1/2 GGML threads and reference mode disabled. Otherwise existing GGML scheduler remains unchanged.
- `GGML_VCPU_MICRO_TRACE=1` emits a one-shot `VLANIX_MICRO_VCPU_DISPATCH units=8 workers=N` for eligible operations. The 8 logical tasks are logical scheduling units, **not** 8 CPU cores.
- A separate diagnostic option `GGML_VCPU_MICRO_STATS=1` records aggregate queue-claim count/ns, worker task time, GGML threadpool barrier elapsed time, and number of worker invocations. These timers **must be disabled in speed runs**.

## Tests and numerical parity

TDD: initial test against the P3.1 build was **RED** because no `VLANIX_MICRO_VCPU_DISPATCH` trace existed. After the implementation the exact trace appeared and native-vs-opt-in parity passed.

Source `tests/test-vcpu-p32-micro8.cpp`: **8/8 PASS** on VPS02:
- 4 representative Qwen-like Q4×Q8 matvec shapes at 2 threads,
- 2 matvec shapes at 1 thread,
- 2 fallback scenarios (2-vector input, matrix with only 64 output rows).
- Max native-vs-opt-in abs error observed `0.000001431` across eligible graph cases.

Source `tests/test-vcpu-p32-logits.cpp`: same local Qwen3.5-0.8B Q4_0, fresh sequential contexts native vs 8-task hook, greedy inference over one synthetic arithmetic prompt:
- **1 worker**: 16/16 token IDs match, 248,320-vocab logit arrays (16 steps), max printed difference 0.000000000, invalid mismatch count zero, PASS.
- **2 workers**: same numerical results, PASS.
- Printed nine digits after decimal; do not interpret as a universal bitwise reproducibility guarantee.

No model weight bytes, prompt output or user data is included in repository evidence.

## Randomized complete Qwen decode benchmark

VPS02 has 2 QEMU vCPUs and only SSE2 baseline. Same lab fork-compiled `llama-bench` and model Qwen3.5-0.8B Q4_0, all CPU, 24 generated tokens, 2 internal repetitions, random profile order (seed 320113); **5 blocks × 4 modes = 20 completed real observations**.

| Mode | Logical Micro-VCPU | OS/GGML workers | Median token/s | Range token/s |
|---|---:|---:|---:|---|
| GGML native | 0 | 2 | 3.2419 | 2.6334–3.8739 |
| P3.2 Micro8 worker1 | 8 | 1 | 3.0258 | 2.6069–3.6758 |
| P3.2 Micro8 worker2 | 8 | 2 | **4.1352** | 3.7612–4.5213 |
| P3.0 direct SSE2 hook without Micro8 task split | 0 | 2 | **4.2486** | 4.1864–5.3695 |

Paired comparison:
- 8 tasks / 2 workers versus GGML native / 2 workers: median paired ratio **1.2605×**, 5/5 paired wins.
- 8 tasks / 2 workers versus prior SSE2 hook / 2 workers: median paired ratio **0.9407×**, only 1/5 wins. Thus eight task shards have **NOT** shown incremental benefit over the existing GGML SSE2 hook; simply splitting more does not give more compute capacity.
- 8 tasks / 1 worker is slower than native 2-worker median, an expected but **not** apples-to-apples worker-count comparison.
- Five paired observations on a noisy shared VPS are preliminary; do not generalize to other CPUs, models, prompts or production loads.

Full evidence: `docs/lab-results/vps02-p32-qwen-micro8-decode-20261113.csv`; validated summary: `docs/lab-results/vps02-p32-qwen-decode-analysis-20261113.json`.

**Build provenance:** `llama-bench` executable was previously built on the same fork and reports `build_commit=99c5d30`; it dynamically links the P3.2 recompiled `libggml-cpu.so` in the isolated lab build directory (early P3.2 scheduling commit `03f92f0` for the randomized measurements). Later P3.2 changes added *opt-in-only* diagnostics, verified by build and graph tests but **the 20 token/s observations were not repeated after instrumentation was added**. This distinction is intentional.

## Real model synchronization and work-queue diagnostics

The diagnostic runs are **separate from the 20 speed observations**: Qwen 8 generated tokens, one repetition, `GGML_VCPU_MICRO_STATS=1`. Each covers **1,350 eligible matvec operator executions × 8 logical tasks = 10,800 actual work units**.

| Instrumented metric | 1 worker | 2 workers |
|---|---:|---:|
| Atomic task claims | 10,800 | 10,800 |
| Mean measured atomic claim time | **53.75 ns** | **308.66 ns** |
| Mean measured GGML barrier elapsed time per worker arrival | **46.28 ns** | **5,639.30 ns** |
| Sum task computation time across workers | 2.231 s | 2.799 s |
| Sum queue claim ns | 580,485 ns | 3,333,573 ns |
| Sum barrier ns | 62,477 ns | 15,226,107 ns |

Source: `docs/lab-results/vps02-p32-micro8-synchronization-20261113.json`.

**Critical caveats:** These are instrumented aggregate values and not unbiased measures of the synchronization penalty in end-to-end decode. Barrier elapsed time **includes worker arrival skew and quantization work imbalance**; summed worker task durations overlap and cannot be added directly to wall time. Do not infer a 2-worker wall-time loss by summing these values. The diagnostics highlight a candidate optimization target (queue contention / load balance), not proof of causation.

## Reproduction on authorized VPS02 lab

```bash
P=/root/vcpu-micro-bench-p0/p3
# The same source/version and local GGML library build; no production library or service replacement.
GGML_VCPU_LAB_ENABLE=1 GGML_VCPU_MICRO_ENABLE=1 \
GGML_VCPU_MICRO_TASKS=8 GGML_VCPU_MICRO_TRACE=1 \
"$P/test-vcpu-p32-micro8"

GGML_VCPU_LAB_ENABLE=1 GGML_VCPU_MICRO_ENABLE=1 \
GGML_VCPU_MICRO_TASKS=8 GGML_VCPU_MICRO_TRACE=1 \
"$P/test-vcpu-p32-micro8" --one-worker-only

python3 scripts/vcpu-p32-validate.py --test
python3 scripts/vcpu-p32-validate.py \
  docs/lab-results/vps02-p32-qwen-micro8-decode-20261113.csv
```

For whole-model decode, use the lab-built `llama-bench` with `-m Qwen3.5-0.8B-Q4_0.gguf --offline -p 0 -n 24 -t 1|2 -r 2 -ngl 0 -o csv`, vary only the opt-in flags and worker count; use `scripts/vcpu-p32-qwen-sweep.py` to randomize and verify hook traces.

## Conclusion and next research gate

**P3.2 executes correctly and shows benefit over native, but the fixed eight-shard scheduler is not faster than the P3.0 two-worker SSE2 kernel in current data.** Next step should focus on adaptive chunk count / row block size, reducing atomic-queue contention, improving locality and eliminating redundant synchronization; test within the *same actual GGML graph*, with the baseline hook without extra partitioning as the true optimization control. Do not add more physical worker threads beyond actual core availability. A second VPS or RPC-based distribution should remain deferred.

All code/data are lab-only and intended for a **Draft PR**. No merge, running production model, system libraries, WireGuard, or services were modified.
