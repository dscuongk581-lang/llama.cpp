# VLANIX P3.1 — Deterministic Qwen output, token ID and logits parity

Date: 2026-11-13 (Asia/Ho_Chi_Minh). VPS02 isolated research environment.

## Objective

P3.0 integrated a Q4_0×Q8_0 SSE2 dot hook into the **real GGML CPU MUL_MAT** path of the fork, compiled **OFF by default**, runtime **OFF by default**. P3.0 measured Qwen3.5-0.8B full-decode throughput, using the same fork-built `llama-bench`: 6 randomly ordered pairs of 24-token tests, two benchmark internal repetitions each, at 2 threads, Q4_0 model, 0 GPU layers.

- GGML native median **3.2626 tok/s**.
- Hook enabled median **4.8405 tok/s**.
- Median paired speed ratio **1.4597×**; hook faster in 6/6 paired repetitions.
- This is **SSE2 lab machine** evidence, not proof on AVX2, ARM, or across multiple VPS nodes.
- P3.0 did not yet establish output token correctness.

P3.1 verifies that optimizing the CPU operator does not change Qwen's generated content in several deterministic prompts, then compares greedy **token IDs and full logits vectors directly through llama_decode**.

## 1. Independent CLI output comparisons

Script: `scripts/vcpu-p31-cli-parity.py`. Uses same separate `build-p30/bin/llama-cli` binary and existing local Qwen3.5-0.8B Q4_0 GGUF, two CPU threads, CPU-only, temperature 0, fixed seed per case, 32 tokens.

- Three synthetic prompts: arithmetic sequence, Python function generation and short factual explanation.
- Sequential native and hook runs per prompt; `GGML_VCPU_LAB_TRACE=1` confirms the hook only for enabled runs.
- Compare generated text **after stripping ANSI escape sequences and the CLI timing footer** (different throughput numbers must not cause false differences). The parser fails closed if expected generated-text markers or timing footer are absent.
- Results: **3/3 generated-text byte-for-byte matches**; same SHA256 per native/hook case.
- Parser unit tests: **3/3 PASS** on VPS02. CI includes these tests.

Only the metadata and generated-text SHA256 checksums are committed, no model weights or prompt outputs:
`docs/lab-results/vps02-p31-qwen-cli-text-parity-20261113.json`.

**CLI limitation:** identical displayed text can mask distinct token IDs/logits; this is why a direct C++ test is included.

## 2. Direct model-level logits and greedy token-ID comparison

Source: `tests/test-vcpu-p31-logits.cpp`. Linked against the **same fork-built** libllama/GGML libraries, not the older b12314 installation.

- Loads the same model **once**, creates two **sequential, fresh** llama_context instances using the same parameters.
- Runs the same synthetic arithmetic prompt under native and SSE2 hook, collecting full F32 logits at every step, selecting the argmax greedily.
- Compares **16 generated greedy token IDs** and **16 × 248,320 vocabulary logits** (approx. 3.97 million pairs).
- The real GGML hook trace `VLANIX_VCPU_LAB_HOOK_HIT` is present.
- Observed: `token_ids_match=true`, **16/16 tokens matched**, maximum printed absolute logit difference `0.000000000`, mean printed absolute logit difference `0.000000000`, zero invalid nonfinite mismatches, PASS.
- Log prints nine decimal places; treat this as **no difference detected at printed precision** rather than proof that every possible run is bit-identical.

Structured result: `docs/lab-results/vps02-p31-qwen-logits-parity-20261113.json`. Test binary executed only on VPS02; CI compilation checks syntax without requiring or uploading Qwen model weights.

### Reproducibility

```bash
# In the P3 fork source with the P3.0 ON lab build in build-p30:
python3 scripts/vcpu-p31-cli-parity.py --test
python3 scripts/vcpu-p31-cli-parity.py \
  --cli build-p30/bin/llama-cli \
  --model /root/vlanix-qwen35-lab/Qwen3.5-0.8B-Q4_0.gguf \
  --output /root/vcpu-micro-bench-p0/p3/p31_qwen_text_parity.json \
  --tokens 32

g++ -std=c++17 -O2 -pthread -I include -I ggml/include \
  tests/test-vcpu-p31-logits.cpp -o build-p30/test-vcpu-p31-logits \
  -L build-p30/bin -lllama -lggml-cpu -lggml-base \
  -Wl,-rpath,"$PWD/build-p30/bin"
GGML_VCPU_LAB_TRACE=1 build-p30/test-vcpu-p31-logits \
  /root/vlanix-qwen35-lab/Qwen3.5-0.8B-Q4_0.gguf
```

Do not run this against a different GGML ABI or replace existing VPS02 Qwen binaries/libraries.

## Safety boundaries and next investigation

- P3.1 is a **short deterministic parity check**, not exhaustive accuracy certification. Greedy and CLI tests do not cover stochastic sampling, multiple long prompts, long-context recurrence, all quantized weights, prefill/batching corner cases, or divergent hardware.
- P3.0 SSE2 speedup and P3.1 correctness are both from the same two-vCPU QEMU host, not a controlled multi-host trial.
- P3.0 CPU operation hook is not yet a full VLANIX backend device or distributed Micro-VCPU scheduler. No proof exists for cross-node distributed inference acceleration.
- GGML CPU remains default unless compiled opt-in and explicitly enabled at runtime. **Keep PR Draft**.
- P3.2 recommended: expand direct logits checks to 128+ tokens, multiple prompt lengths and prefill, gather per-op timing/counters for effective portion of inference, compare reproducible native/hook runs on second CPU/ISA, then design VLANIX backend contract.

**Status:** Three deterministic generated-text parity cases PASS; direct greedy/token/logit parity PASS for one 16-token case; repository and CI checks added without model data or production changes.
