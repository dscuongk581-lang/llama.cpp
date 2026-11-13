// VLANIX P3.0: actual GGML graph parity with the opt-in Q4_0 x Q8_0 CPU hook.
// Uses deterministic synthetic tensors only. No model or prompt data is needed.
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void compute_case(int rows, int cols, int vectors, int threads) {
    if (cols % 32 != 0) throw std::runtime_error("unaligned rows");
    ggml_init_params params = {};
    params.mem_size = 64 * 1024 * 1024;
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("ggml_init failed");

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, cols, rows);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, vectors);
    std::vector<float> weights(size_t(rows) * cols);
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = float(int(i % 97) - 48) / 97.0f;
    size_t bytes = ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(),
                                       w->data, 0, rows, cols, nullptr);
    if (bytes != ggml_nbytes(w)) throw std::runtime_error("Q4_0 byte mismatch");
    for (int i = 0; i < cols * vectors; ++i)
        ggml_get_data_f32(x)[i] = float((i % 41) - 20) / 41.0f;

    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    if (setenv("GGML_VCPU_LAB_ENABLE", "0", 1) != 0)
        throw std::runtime_error("setenv off failed");
    if (ggml_graph_compute_with_ctx(ctx, graph, threads) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("native graph failed");
    auto * result = ggml_get_data_f32(y);
    std::vector<float> baseline(result, result + size_t(rows) * vectors);
    if (setenv("GGML_VCPU_LAB_ENABLE", "1", 1) != 0)
        throw std::runtime_error("setenv on failed");
    if (ggml_graph_compute_with_ctx(ctx, graph, threads) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("opt-in graph failed");
    double max_error = 0;
    for (size_t i = 0; i < baseline.size(); ++i) {
        if (!std::isfinite(result[i])) throw std::runtime_error("nonfinite output");
        max_error = std::max(max_error, std::abs(double(result[i]) - baseline[i]));
    }
    if (max_error > 0.002) {
        std::fprintf(stderr, "VCPU_GRAPH_MISMATCH rows=%d cols=%d vectors=%d error=%.9f\n",
                     rows, cols, vectors, max_error);
        throw std::runtime_error("output parity failed");
    }
    std::printf("VCPU_GRAPH_PARITY_PASS rows=%d cols=%d vectors=%d threads=%d max_abs_error=%.9f\n",
                rows, cols, vectors, threads, max_error);
    ggml_free(ctx);
}

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--one-worker-only") == 0) {
            compute_case(512, 1024, 1, 1);
            compute_case(3584, 1024, 1, 1);
            std::puts("P32_ONE_WORKER_PASS 2/2");
            return 0;
        }
        // Single-token decode candidates: hook may be active when compiled/enabled.
        compute_case(512, 1024, 1, 2);
        compute_case(1024, 2048, 1, 2);
        compute_case(3584, 1024, 1, 2);
        compute_case(1024, 3584, 1, 2);
        // Same 8 logical Micro-VCPUs, but only one physical worker.
        compute_case(512, 1024, 1, 1);
        compute_case(3584, 1024, 1, 1);
        // Multi-token and sub-128-row shapes must remain safely executable.
        compute_case(512, 1024, 2, 2);
        compute_case(64, 1024, 1, 2);
        std::puts("P32_GRAPH_TESTS_PASS 8/8");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "P32_GRAPH_TEST_ERROR: %s\n",e.what());
        return 1;
    }
}
