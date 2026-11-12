# VLANIX P3.0 — Real GGML CPU Q4×Q8 operator hook, VPS02

Date: 2026-11-13 (Vietnam). Fork: `dscuongk581-lang/llama.cpp`, base commit `8ca5a8008b57aa39f2601546c0d07912045080a5`.

**Scope:** a narrowly scoped, **compile-time OFF-by-default / runtime OFF-by-default** Q4_0×Q8_0 SSE2 dot hook inside the actual GGML CPU `GGML_OP_MUL_MAT` path. This is not yet a VLANIX network/distributed backend or the final VCPU scheduler.

## Why this matters

P2.0–P2.8 measured external experimental Micro-VCPU executables that consumed GGML tensor buffers. Those executables did not participate in actual `ggml_graph_compute` or `llama_decode`. This change invokes the experimental dot directly from `ggml_compute_forward_mul_mat_one_chunk`, which is reached from `ggml_backend_cpu_graph_compute` and the normal GGML graph execution pipeline. The real hook was confirmed by the one-shot `VLANIX_VCPU_LAB_HOOK_HIT` trace in a GGML graph on VPS02.

## What changed

1. `ggml/src/ggml-cpu/vcpu-lab-q4q8.h`: SSE2 Q4_0×Q8_0 dot computes signed-nibble/int8 products with `pmaddwd` and FP16 per-block scale, matching GGML Q4_0 block layout. Uses a **single dot worker**, not a new nested threadpool.
2. `ggml/src/ggml-cpu/ggml-cpu.c`: for eligible `MUL_MAT` Q4_0×Q8_0 calls, `GGML_VCPU_LAB_ENABLE=1` allows the SSE2 dot. Otherwise the **original native `vec_dot` is unchanged**.
3. `ggml/src/ggml-cpu/CMakeLists.txt`: option `GGML_VCPU_LAB=OFF` (default). Compilation requires explicit opt-in and x86 SSE2 architecture.
4. `tests/test-vcpu-lab-mul-mat.cpp`: real GGML graph correctness test across six cases, including unsupported multi-vector and small matrix fallback.
5. `.github/workflows/vlanix-vcpu-lab.yml`: separate CI build with feature ON/OFF, graph correctness and hook/no-hook tracing, no model file.
6. `scripts/vcpu-lab-evidence.py`: validates 96 randomized real-weight graph observations, per-pair digest and numerical accuracy. Raw result CSV contains aggregate timings and digests **only**.

### Guard

The hook is available only when all apply:
- build `GGML_VCPU_LAB=ON`
- runtime `GGML_VCPU_LAB_ENABLE=1`
- target is x86 with SSE2 at compile time
- current CPU op has Q4_0 weights, GGML-selected Q8_0 dot operands, F32 activation input, a single vector and no reference-mode calculation
- columns divisible by 32; weight rows >=128; one logical batch in dimensions 2/3
- GGML original CPU graph scheduling, allocation, and threadpool are preserved

All other cases call original GGML `vec_dot`. The first eligible call can optionally emit trace by setting `GGML_VCPU_LAB_TRACE=1`. The optional trace is intentionally one-shot to minimize logging overhead.

**Safety:** It is a lab-only feature. It must not be used for production inference without broad numerical/determinism, regression, power and whole-model tests. This hook does not add a proper GGML backend device, and it does not perform distributed scheduling or remote VCPU execution.

## Reproduced build on VPS02

A separate shallow clone of the fork was placed at `/root/vcpu-micro-bench-p0/p3/llama-fork` and compiled at pinned branch SHA. To avoid changing system software, CMake 3.25 and required libraries were downloaded/extracted into a **local lab-only directory**. GCC 12 toolchain; 2 QEMU vCPUs.

The QEMU CPU exposes SSE2 but not AVX2; all AVX/FMA/BMI2/SSE4.2 options were explicitly disabled to ensure an x86_64 SSE2-compatible baseline.

```bash
cmake -S . -B build-p30 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VCPU_LAB=ON \
  -DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF \
  -DGGML_SSE42=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build-p30 --target ggml-cpu -j2
```

**Result:** local lab build PASS. A second complete `GGML_VCPU_LAB=OFF` CPU backend build also PASS, 6/6 parity tests PASS, and setting runtime `GGML_VCPU_LAB_ENABLE=1` did **not** trigger the hook in that OFF build. No installed, system-wide or live Qwen binary/library was replaced.

Actual graph parity:
- 512×1024 single vector: max native/hook absolute difference 0.000000477
- 1024×2048 single vector: 0.000000954
- 3584×1024 single vector: 0.000000477
- 1024×3584 single vector: 0.000001431
- 512×1024 two vectors (fallback): 0
- 64×1024 single vector (small-operator fallback): 0

**6/6 PASS.** One-shot hook log confirmed in eligible case.

## Real Qwen packed weight operator benchmark

Uses four **actual Q4_0 weight matrices extracted read-only from** `Qwen3.5-0.8B-Q4_0.gguf`, kept private on VPS02. **Activation vector remains deterministic synthetic F32**, which GGML quantizes to Q8_0 in its standard computation. No model weights or activation dumps are committed.

Same **compiled GGML CPU library**, same 2-thread configuration, same graph operator and weight bytes, toggling only `GGML_VCPU_LAB_ENABLE`. Each trial computes 40 matrix–vector operations. Randomized order, seed 300113; 12 repetitions per route and tensor = 96 total samples, all passed numeric tolerance 0.002 and digest parity.

| Qwen tensor | Rows×cols | Native median 40-ops (ms) | Lab hook median 40-ops (ms) | Median latency ratio native/hook |
|---|---|---:|---:|---:|
| attn_k | 512×1024 | 18.889 | **11.264** | 1.677× |
| ssm_out | 1024×2048 | 55.663 | **28.703** | 1.939× |
| ffn_gate | 3584×1024 | 88.110 | **50.218** | 1.755× |
| ffn_down | 1024×3584 | 96.295 | **45.225** | 2.129× |

The per-repetition paired 8-train/4-holdout evaluator reported lower bootstrap ratios between roughly 1.50 and 1.95 and hook wins in 4/4 holdout repetitions for all four shapes. **This is operator-level evidence in one shared SSE2-only VPS session**, not transferable to different CPUs, newer native ISA kernels, workloads or full-model token throughput without validation.

Raw source: `docs/lab-results/vps02-p30-ggml-graph-qwen-weights-20261113.csv`.
Summary: `docs/lab-results/vps02-p30-ggml-graph-evidence-20261113.json`.

## Version and integration boundary

- Fork base commit `8ca5a8008b57aa39f2601546c0d07912045080a5` was pinned and built in an isolated lab directory.
- The preexisting VPS02 `llama-cli` binary was build b12314 commit `e1f7593c3` (an older **different code revision**). Neither its binaries nor its shared libraries were changed.
- No plugin or shared library built against the newer fork was loaded into that older executable; thus no cross-version ABI gamble.
- No evidence yet of complete Qwen `llama_decode` throughput or token correctness from this hook, unless explicitly added with separate model-level measurements. Do not extrapolate operator speedups directly into token/s.

## Proposed next gates

1. Build a **lab-only llama-bench/llama-cli from this exact fork source**, then run native-vs-opt-in identical Qwen decode tests, keeping prompt, context, thread count and CPU affinity fixed.
2. Capture optional aggregate counts of eligible/hit operators (not user prompts) and check percentage of total decode time actually spent in this kernel.
3. Validate larger sets of real activation vectors and model outputs; compare temperature=0 token sequences and logit tolerances.
4. Check multi-token/prefill, graph reuse, other quantization types, parallel batch, dynamic backend loading, and platforms without SSE2; keep native fallback.
5. After acceptable whole-decode correctness and speed evidence, explore a proper pluggable VLANIX GGML backend. Remote/distributed VCPU remains a separate later step.

**Status:** P3.0 true GGML graph CPU operator hook PASS in VPS02 lab; deployment and full-model conclusions remain blocked. PR must stay Draft.
