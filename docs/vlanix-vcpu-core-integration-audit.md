# VLANIX / VCPU — llama.cpp Core Integration Audit

**Date:** 2026-11-13. **Scope:** read-only architecture examination of this fork + existing P2.0–P2.8 experiments on VPS02. This document does **not** enable any custom kernel, deploy backend libraries, or alter inference.

## Executive assessment

**Hypothesis:** "The llama.cpp wrapper conflicts with VCPU."  
**Finding:** **Plausible at the integration boundary, but not established as a runtime conflict.** In the examined code, `llama.cpp` builds a GGML graph and uses GGML's backend scheduler to compute it. Previously measured VLANIX Micro-VCPU kernels are separate C++ test executables that *read GGML buffers* but **are not registered GGML backends or invoked through llama_decode**. Hence no evidence of a failed direct integration exists; the bridge has not yet been implemented.

The bottleneck is likely a combination of **missing integration**, **native GGML's specialized Q4×Q8 kernels**, **threadpool and quantization semantics**, and **inter-backend scheduling/copy overhead** if offloading is attempted. This should be measured per operation instead of described simply as a wrapper conflict.

## Proven code path (fork master at audit time)

1. `src/llama-graph.cpp`: creates `ggml_mul_mat` nodes (e.g. inside model graph builders).
2. `src/llama-context.cpp`:
   - `llama_context::process_ubatch` at approx. line 1817 builds/allocates/reuses GGML graph.
   - `llama_context::graph_compute` at approx. line 3123 configures `n_threads` and the CPU threadpool then calls `ggml_backend_sched_graph_compute_async`.
3. `ggml/src/ggml-backend.cpp`:
   - `ggml_backend_sched_split_graph` at approx. line 1483 chooses per-node device assignment/splits.
   - `ggml_backend_sched_compute_splits` at approx. line 2210 synchronizes backends and copies tensor inputs where required.
   - `ggml_backend_sched_graph_compute_async` at approx. line 2467 allocates graph if needed and computes splits.
4. `ggml/src/ggml-cpu/ggml-cpu.cpp`:
   - `ggml_backend_cpu_device_supports_op` at approx. line 424 advertises supported operations.
   - `ggml_backend_cpu_graph_compute` at approx. line 170 plans `ggml_graph_plan` and calls `ggml_graph_compute`.
5. `ggml/src/ggml-cpu/ggml-cpu.c`:
   - `ggml_graph_compute` at approx. line 3400 starts/reuses threadpool.
   - `GGML_OP_MUL_MAT` dispatch calls `ggml_compute_forward_mul_mat`.
   - `ggml_compute_forward_mul_mat` at approx. line 1264 checks for specialized tiled path; converts activation into expected `vec_dot_type` if needed.
   - `ggml_compute_forward_mul_mat_one_chunk` at approx. line 1174 obtains `type_traits_cpu[type].vec_dot` and processes tiled row ranges.
6. `ggml/src/ggml-cpu/quants.c`: provides generic `ggml_vec_dot_q4_0_q8_0_generic`; CPU-specific SIMD implementations may be chosen by ISA-specific traits/backends.

**Where does VCPU belong?** The most economical initial integration point is **a narrow, opt-in Q4×Q8 matrix-vector operation hook within the GGML CPU implementation** (without changing graph construction), followed by a **proper backend plugin** only if it proves worthwhile. Changes at `llama_decode` or tokenizer/API levels would be premature.

## Critical environment mismatch

- The fork was confirmed through `git ls-remote` at **`8ca5a8008b57aa39f2601546c0d07912045080a5`**, equal to upstream's `HEAD` at examination: no VLANIX-specific kernel/backend changes on the default branch.
- The VPS02 binary actually used in P2.0–P2.8 is llama.cpp **build 12314**, commit **`e1f7593c3`** (reported by `llama-cli --version`). It is **not** the same commit as the fork's current `master`.
- Never assume the ABI, struct layouts, quantization traits, backend registration or plugin API are identical between these revisions. Do not hot-load a plugin compiled against newer headers into the older process without version/ABI verification.
- VPS02 has **2 QEMU virtual CPUs and SSE2**; it does not advertise AVX2. Any optimization requiring newer ISA is ineligible there.

## What existing VLANIX experiments actually proved

P2.3 revealed why naive Q4×F32 arithmetic differed from GGML's Q4×Q8 path: explicit Q8 quantization of activations reduced native-vs-reference absolute discrepancy from ~0.01420 to ~0.00000129 in one test. P2.4 SSE2 improved some isolated microbenchmarks, but P2.5–P2.8 show inconsistent speedups across repetitions and shapes, including negative results. P2.8 used **real Qwen GGUF Q4_0 weights**, but the activation vector remained synthetic. There has been **no Qwen decoding through a Micro-VCPU GGML backend**, and no proven end-to-end token/s gain.

These results do not demonstrate an incompatibility bug inside llama.cpp. They establish numerical and performance requirements for a real integration.

## Candidate integration paths (in priority order)

### A. GGML CPU operation hook — **first proof-of-concept**
- Build a **separate experimental llama.cpp binary** from one pinned commit (prefer the fork and match corresponding headers/libs), with `GGML_VCPU_LAB=ON` or equivalent compile-time feature disabled by default.
- Restrict to `GGML_OP_MUL_MAT` with contiguous Q4_0 weights + F32 activation, decode-specific matrix-vector shape and supported SSE2 CPU; verify precisely the `vec_dot_type` and quantization behaviors.
- Run an **explicit off-by-default lab dispatcher** choosing the optimized Micro-VCPU kernel only if offline cost evidence and a numeric canary pass. On any unsupported geometry, precision mismatch, device unsupported, or speed uncertainty, use existing GGML native compute path.
- Keep pool lifetime persistent; avoid constructing worker threads for each tensor operation. Make sure no nested threadpool oversubscription with GGML's own worker threads.
- Test tensor byte strides, shapes, batch sizes, multi-token prompt processing, reuse of per-operation Q8 quantization, and per-layer error propagation.

### B. GGML device/backend plugin — **second stage**
- `ggml/src/ggml-backend-reg.cpp` supports loading backends; `ggml/include/ggml-backend.h` exposes backend registration/device interfaces, plus explicit backend scheduling controls.
- Implement device `supports_op`, host buffer semantics, allocation, transfer, synchronization, `graph_compute`, and fallback; do **not** report broad operator support until tests exist.
- A plugin cannot simply intercept an individual CPU kernel because GGML scheduler **assigns backend per node/split**, handles buffer placement, and may copy tensors; overly narrow support can cause expensive split/copy operations.
- Pin the exact ABI before building or loading such a plugin and benchmark graph-level speed (including allocations and copies).

### C. Network/VCPU distributed execution — **only for coarse work**
- Reuse the actual `GGML` backend device and/or `ggml-rpc` mechanisms as a design reference, but do not assume RPC overhead is acceptable for per-token microkernels.
- Keep small, latency-sensitive matvec on the local native backend unless proven otherwise; use larger independent tensors/layers or batched work for remote compute.
- Measure transfer bytes, KV/cache locality, synchronization, retries, deterministic checksums and loss of a node before claiming benefit.

## P3.0 executable acceptance gates (proposed)

1. **Version gate:** build a lab-only fork at a pinned commit; record version/hash of `ggml.h`, `ggml-backend.h`, CPU kernel source and resulting `llama-cli`; keep VPS02 b12314 baseline for comparison but **do not mix their libraries**.
2. **Instrumentation gate:** profile graph-level `GGML_OP_MUL_MAT` / `MUL_MAT_ID` call counts, Q4_0/Q8_0 conversion, kernel CPU time, threadpool barriers, scheduler tensor copies, and actual decode time on Qwen; log only aggregates and avoid publishing user content.
3. **Correctness gate:** verify native vs opt-in outputs on multiple Qwen weight matrices *and actual activation vectors* captured from a consented lab run; compare layer output norms, max/relative error and identical prompt/seed outputs.
4. **Speed gate:** paired randomized native-vs-custom GGML graph inference runs across multiple prompts, batch sizes and shapes, with cold-start separately accounted. Require a positive heldout lower confidence bound **after** Q8 conversion, pool dispatch and buffer/copy costs; otherwise native default.
5. **Resilience gate:** unsupported operations, SIMD missing, graph reallocation, backend unload, buffer ownership and remote-node failure must preserve native fallback without silent corruption.

## Decision

**Do not rewrite the whole llama.cpp wrapper.** Begin with a pinned-source GGML CPU operation hook in a separate lab build, and only promote to a custom backend after demonstrating numeric parity and measurable gain inside an actual GGML graph. A distributed VCPU backend is a later concern, because the scheduler/copy/latency structure, not the language-model API, is the relevant constraint.

All findings were obtained without changing the running model, installed GGML libraries or production configuration.
