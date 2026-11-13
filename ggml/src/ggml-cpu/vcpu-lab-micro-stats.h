// VLANIX P3.2: opt-in diagnostic counters for GGML's existing atomic
// micro-task claim queue. A diagnostic run is NOT a timing performance run.
#pragma once
#if defined(GGML_VCPU_LAB) && defined(__SSE2__) && (defined(__x86_64__) || defined(__i386__))
#include <stdint.h>
#include <stdatomic.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

static _Atomic uint64_t vcpu_micro_work_units = 0;
static _Atomic uint64_t vcpu_micro_work_ns = 0;
static _Atomic uint64_t vcpu_micro_atomic_claims = 0;
static _Atomic uint64_t vcpu_micro_atomic_ns = 0;
static _Atomic uint64_t vcpu_micro_worker_visits = 0;
static _Atomic uint64_t vcpu_micro_barrier_ns = 0;
static _Atomic uint64_t vcpu_micro_barrier_visits = 0;
static int vcpu_micro_stats_initialized = 0;

static inline uint64_t vcpu_micro_monotonic_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

static void vcpu_micro_stats_print(void) {
    const uint64_t units = atomic_load(&vcpu_micro_work_units);
    const uint64_t claims = atomic_load(&vcpu_micro_atomic_claims);
    const uint64_t work = atomic_load(&vcpu_micro_work_ns);
    const uint64_t atomic_time = atomic_load(&vcpu_micro_atomic_ns);
    const uint64_t visits = atomic_load(&vcpu_micro_worker_visits);
    const uint64_t barrier = atomic_load(&vcpu_micro_barrier_ns);
    const uint64_t barrier_visits = atomic_load(&vcpu_micro_barrier_visits);
    fprintf(stderr,
            "VLANIX_MICRO_VCPU_STATS units=%llu claims=%llu claim_ns=%llu work_ns=%llu worker_visits=%llu barrier_ns=%llu barrier_visits=%llu note=diagnostic_instrumented_barrier_includes_arrival_skew\n",
            (unsigned long long)units, (unsigned long long)claims,
            (unsigned long long)atomic_time, (unsigned long long)work,
            (unsigned long long)visits,
            (unsigned long long)barrier, (unsigned long long)barrier_visits);
}

static inline void vcpu_micro_stats_register(void) {
    if (__sync_bool_compare_and_swap(&vcpu_micro_stats_initialized,0,1)) {
        atexit(vcpu_micro_stats_print);
    }
}
#endif
