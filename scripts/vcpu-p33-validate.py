#!/usr/bin/env python3
"""P3.3 complete paired Qwen decode evidence guard and descriptive comparison.

Five randomized repeats x five profiles; checks trace/worker configuration.
No automatic promotion from these small noisy experiments.
"""
import csv, math, statistics, json, sys, unittest
from collections import Counter
EXPECTED={"native2":(2,"0",0,0),"hook2":(2,"0",1,0),
          "fixed8_2":(2,"8",1,1),"adaptive2":(2,"adaptive",1,1),
          "adaptive1":(1,"adaptive",1,1)}
REPS=5
def validate(rows):
    counts=Counter()
    for r in rows:
        m=r["mode"]
        if m not in EXPECTED:raise ValueError("unknown mode")
        workers,logical,hook,dispatch=EXPECTED[m]
        if (int(r["workers"]),r["logical"],int(r["hook_hit"]),int(r["dispatch_hit"]))!=(workers,logical,hook,dispatch):
            raise ValueError("worker/hook/logical mismatch")
        if int(r["tokens"])!=24:raise ValueError("wrong token count")
        for field in ("avg_ts","stddev_ts"):
            if not math.isfinite(float(r[field])): raise ValueError("nonfinite")
        if float(r["avg_ts"])<=0 or float(r["stddev_ts"])<0:raise ValueError("bad speed")
        if not r["commit"]:raise ValueError("missing build commit")
        repeat=int(r["repeat"])
        if repeat<1 or repeat>REPS:raise ValueError("invalid repetition")
        counts[(repeat,m)]+=1
    expected=Counter({(rep,m):1 for rep in range(1,REPS+1) for m in EXPECTED})
    if counts!=expected:raise ValueError("missing or duplicate observations")
    return {(int(r["repeat"]),r["mode"]):float(r["avg_ts"]) for r in rows}
def evaluate(rows):
    obs=validate(rows)
    stats=[]
    for m,(workers,logical,*_) in EXPECTED.items():
        values=[obs[(r,m)] for r in range(1,REPS+1)]
        stats.append(dict(mode=m,workers=workers,logical_tasks=logical,
            median_tokens_per_sec=round(statistics.median(values),4),
            minimum_tokens_per_sec=round(min(values),4),
            maximum_tokens_per_sec=round(max(values),4)))
    paired={}
    for mode in ("native2","hook2","fixed8_2"):
        ratios=[obs[(r,"adaptive2")]/obs[(r,mode)] for r in range(1,REPS+1)]
        paired[mode]=dict(median_adaptive2_speed_ratio=round(statistics.median(ratios),4),
            adaptive2_wins=sum(x>1 for x in ratios),
            ratios=[round(x,4) for x in ratios])
    return dict(status="lab_only",observations=len(rows),randomized_repeats=REPS,
        qwen="Qwen3.5-0.8B Q4_0",cpu="VPS02 2 QEMU vCPU SSE2",
        profiles=stats,adaptive2_paired_vs=paired,
        decision="NOT_AUTO_PROMOTED",caveat="Only 5 pairs, host noise; no inter-node distribution.")
class Tests(unittest.TestCase):
    def test_missing(self):
        with self.assertRaises(ValueError):validate([])
    def test_unknown(self):
        with self.assertRaises(ValueError):validate([dict(mode="bad")])
    def test_synthetic_complete(self):
        rows=[]
        for r in range(1,REPS+1):
            for m,(workers,logical,h,d) in EXPECTED.items():
                rows.append(dict(repeat=r,mode=m,workers=workers,logical=logical,hook_hit=h,dispatch_hit=d,
                  tokens=24,avg_ts="4.5",stddev_ts="0.1",commit="p33"))
        self.assertEqual(evaluate(rows)["observations"],25)
        with self.assertRaises(ValueError):validate(rows[:-1])
if __name__=="__main__":
    if len(sys.argv)==2 and sys.argv[1]=="--test":
        unittest.main(argv=["p33-validate"],exit=False)
    elif len(sys.argv)==2:
        with open(sys.argv[1],newline="") as file: rows=list(csv.DictReader(file))
        print(json.dumps(evaluate(rows),indent=2,sort_keys=True))
    else:sys.exit("usage: validator --test | path.csv")
