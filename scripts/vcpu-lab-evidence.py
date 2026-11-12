#!/usr/bin/env python3
"""VLANIX P3.0: validate real Qwen GGUF-weight GGML graph operator evidence.
This does NOT authorize llama_decode rollout or claims about token/s.
"""
import argparse
import csv
import json
import math
import random
import statistics
import unittest
from collections import Counter

SHAPES = {
    "attn_k": (512, 1024),
    "ssm_out": (1024, 2048),
    "ffn_gate": (3584, 1024),
    "ffn_down": (1024, 3584),
}
ROUTES = ("native", "vcpu_hook")
N_REPEATS = 12

def validate(rows):
    expected = Counter({(rep, name, route): 1 for rep in range(1, N_REPEATS + 1)
                        for name in SHAPES for route in ROUTES})
    counts = Counter()
    for r in rows:
        name = r["tensor"]
        route = r["route"]
        rep = int(r["repeat"])
        if name not in SHAPES or route not in ROUTES or rep < 1 or rep > N_REPEATS:
            raise ValueError("unexpected tensor/route/rep")
        if (int(r["rows"]), int(r["cols"])) != SHAPES[name]:
            raise ValueError("wrong tensor shape")
        if (r["mode"], r["quant"]) != ("ggml", "Q4_0xQ8_0"):
            raise ValueError("not a GGML Q4xQ8 graph")
        if (int(r["threads"]), int(r["rounds"])) != (2, 40):
            raise ValueError("wrong execution parameters")
        for key in ("seconds", "max_abs_error", "digest"):
            if not math.isfinite(float(r[key])):
                raise ValueError("nonfinite result")
        if float(r["seconds"]) <= 0 or float(r["max_abs_error"]) >= 0.002:
            raise ValueError("latency/accuracy invalid")
        counts[(rep, name, route)] += 1
    if counts != expected:
        raise ValueError("missing or duplicate measurement")
    data = {(int(r["repeat"]), r["tensor"], r["route"]): r for r in rows}
    for rep in range(1, N_REPEATS + 1):
        for name in SHAPES:
            a = float(data[(rep, name, "native")]["digest"])
            b = float(data[(rep, name, "vcpu_hook")]["digest"])
            if abs(a-b) > 0.002:
                raise ValueError("native-vcpu digest drift")
    return data

def bootstrap_low(values):
    rng = random.Random(300113)
    n = len(values)
    medians = sorted(statistics.median(rng.choices(values, k=n)) for _ in range(6000))
    return medians[149]

def evaluate(rows):
    data = validate(rows)
    profiles = []
    for name, (m,k) in SHAPES.items():
        paired = [float(data[(rep,name,"native")]["seconds"]) /
                  float(data[(rep,name,"vcpu_hook")]["seconds"])
                  for rep in range(1,N_REPEATS + 1)]
        train, heldout = paired[:8],paired[8:]
        profiles.append({
            "tensor": name,"rows": m,"cols": k,
            "all_median_native_over_hook": round(statistics.median(paired),4),
            "train_paired_median": round(statistics.median(train),4),
            "train_bootstrap_lower": round(bootstrap_low(train),4),
            "holdout_paired_median": round(statistics.median(heldout),4),
            "holdout_hook_wins": sum(x>1 for x in heldout),
            "full_native_median_ms": round(1000*statistics.median(
                float(data[(rep,name,"native")]["seconds"]) for rep in range(1,13)),3),
            "full_hook_median_ms": round(1000*statistics.median(
                float(data[(rep,name,"vcpu_hook")]["seconds"]) for rep in range(1,13)),3),
        })
    return {
        "status": "standalone_real_GGML_graph_operator_not_full_llama_decode",
        "source_commit": "8ca5a8008b57aa39f2601546c0d07912045080a5",
        "build_isa": "x86_64_baseline_SSE2_no_AVX2",
        "native_default": True,
        "real_qwen_q4_weights": True,
        "real_qwen_activations": False,
        "repeats": N_REPEATS,
        "num_observations": len(rows),
        "per_tensor": profiles,
        "no_end_to_end_token_speedup_claim": True,
    }

class ValidationTests(unittest.TestCase):
    def test_missing_rejected(self):
        with self.assertRaises(ValueError):validate([])
    def test_wrong_tensor_rejected(self):
        with self.assertRaises(ValueError):
            validate([dict(repeat="1",tensor="invalid",route="native")])
    def test_bootstrap_deterministic(self):
        self.assertEqual(bootstrap_low([1.1,1.2,0.9,1,1.3,1.4,1.05,1.2]),
                         bootstrap_low([1.1,1.2,0.9,1,1.3,1.4,1.05,1.2]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--test",action="store_true")
    args=parser.parse_args()
    if args.test:
        unittest.main(argv=["vcpu-lab-evidence"],exit=False)
    else:
        if not args.input:parser.error("--input is required")
        with open(args.input,newline="") as handle:rows=list(csv.DictReader(handle))
        print(json.dumps(evaluate(rows),indent=2,sort_keys=True))
