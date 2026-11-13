#!/usr/bin/env python3
"""P3.2 validate seeded paired decode benchmark metadata; no speedup threshold."""
import csv
import json
import math
import statistics
import sys
import unittest
from collections import Counter

REPEATS=5
MODES={
    "native2":(2,0,0,0),
    "micro8_worker1":(1,8,1,1),
    "micro8_worker2":(2,8,1,1),
    "hook2_no_micro":(2,0,1,0),
}
def validate(rows):
    counts=Counter()
    for r in rows:
        mode=r["mode"]
        if mode not in MODES:raise ValueError("unknown mode")
        t,u,hook,micro=MODES[mode]
        rep=int(r["repeat"])
        if not 1<=rep<=REPEATS: raise ValueError("repeat bounds")
        if (int(r["worker_threads"]),int(r["micro_tasks"]),int(r["hook_hit"]),int(r["micro_dispatch_hit"]))!=(t,u,hook,micro):
            raise ValueError("hook/worker mismatch")
        if (int(r["n_gen"]),int(r["n_prompt"]),int(r["bench_repeats"]))!=(24,0,2):
            raise ValueError("wrong benchmark settings")
        val=float(r["tokens_per_sec"])
        if not math.isfinite(val) or val<=0:raise ValueError("bad throughput")
        std=float(r["std_tokens_per_sec"])
        if not math.isfinite(std) or std<0:raise ValueError("bad throughput variance")
        if not r["commit"].strip():raise ValueError("no binary commit")
        counts[(rep,mode)]+=1
    expected=Counter({(rep,mode):1 for rep in range(1,REPEATS+1) for mode in MODES})
    if counts!=expected:raise ValueError("missing or duplicate measurement")
    return {(int(r["repeat"]),r["mode"]):float(r["tokens_per_sec"]) for r in rows}
def analyze(rows):
    x=validate(rows)
    profiles=[]
    for m in MODES:
        scores=[x[(i,m)] for i in range(1,REPEATS+1)]
        profiles.append({"mode":m,"workers":MODES[m][0],
                         "micro_tasks":MODES[m][1],
                         "median_tokens_per_sec":round(statistics.median(scores),4),
                         "minimum_tokens_per_sec":round(min(scores),4),
                         "maximum_tokens_per_sec":round(max(scores),4)})
    ratios={}
    for m in ("micro8_worker1","micro8_worker2","hook2_no_micro"):
        values=[x[(i,m)]/x[(i,"native2")] for i in range(1,REPEATS+1)]
        ratios[m]={"median_paired_speedup_vs_native2":round(statistics.median(values),4),
                   "wins_out_of_5":sum(a>1 for a in values),
                   "per_pair_speedup":[round(a,4) for a in values]}
    between=[x[(i,"micro8_worker2")]/x[(i,"hook2_no_micro")] for i in range(1,REPEATS+1)]
    return {"status":"experimental_single_vps_decode_not_distributed",
            "source":"VPS02 QEMU 2 virtual CPU, SSE2-only fork-built llama-bench",
            "qwen_model":"Qwen3.5-0.8B Q4_0","logical_microvcpu_count":8,
            "repeats":REPEATS,"observations":len(rows),
            "profiles":profiles,"native_relative_paired":ratios,
            "micro8_worker2_vs_hook2_no_micro":{
                "median_paired_ratio":round(statistics.median(between),4),
                "micro8_faster_pairs":sum(a>1 for a in between),
                "per_pair_ratio":[round(a,4) for a in between]},
            "limitations":"No cross-device speedup; shared VPS load, small N, no long-prompt parity measured here."}
class Tests(unittest.TestCase):
    def test_empty_is_invalid(self):
        with self.assertRaises(ValueError):validate([])
    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):validate([{"mode":"invented"}])
    def test_analyze_uses_true_pairs(self):
        rows=[]
        for rep in range(1,REPEATS+1):
            for mode,(threads,tasks,hook,micro) in MODES.items():
                rows.append(dict(repeat=rep,mode=mode,worker_threads=threads,micro_tasks=tasks,
                    hook_hit=hook,micro_dispatch_hit=micro,n_gen=24,n_prompt=0,
                    bench_repeats=2,tokens_per_sec=3 if mode=="native2" else 6,
                    std_tokens_per_sec=0.1,commit="lab"))
        self.assertEqual(analyze(rows)["native_relative_paired"]["micro8_worker2"]["median_paired_speedup_vs_native2"],2.0)
if __name__=="__main__":
    if len(sys.argv)==2 and sys.argv[1]=="--test":
        unittest.main(argv=["p32-validate"],exit=False)
    elif len(sys.argv)==2:
        with open(sys.argv[1],newline="") as f:rows=list(csv.DictReader(f))
        print(json.dumps(analyze(rows),indent=2,sort_keys=True))
    else:
        sys.exit("usage: vcpu-p32-validate.py --test | benchmark.csv")
