#!/usr/bin/env python3
"""P3.5 paired Qwen decode benchmark integrity checker.
Exploratory: NO scheduling policy auto-promotion based on five reps.
"""
import csv,json,math,statistics,sys,unittest
from collections import Counter
MODES={
 "native2":("0","0",0,0,0),
 "hook2":("0","0",1,0,0),
 "adaptive_atomic2":("adaptive","0",1,1,0),
 "adaptive_static2":("adaptive","static",1,1,1),
 "adaptive_hybrid2":("adaptive","hybrid",1,1,1)}
REPS=5
def validate(rows):
 seen=Counter()
 for r in rows:
  mode=r["mode"]
  if mode not in MODES:raise ValueError("unsupported route")
  tasks,sch,h,d,s=MODES[mode]
  if (r["micro_tasks"],r["scheduler"],int(r["hook_hit"]),int(r["dispatch_hit"]),int(r["scheduler_hit"]))!=(tasks,sch,h,d,s):
   raise ValueError("traces or routing mismatch")
  if (int(r["worker_threads"]),int(r["n_gen"]),int(r["bench_repeats"]))!=(2,24,2):raise ValueError("bad test parameters")
  for key in ("tokens_per_sec","stddev_tokens_per_sec"):
   if not math.isfinite(float(r[key])):raise ValueError("nonfinite")
  if float(r["tokens_per_sec"])<=0 or float(r["stddev_tokens_per_sec"])<0:raise ValueError("invalid result")
  if not r["build_commit"].strip():raise ValueError("missing build")
  repeat=int(r["repeat"])
  if repeat not in range(1,REPS+1):raise ValueError("out of bound repeat")
  seen[(repeat,mode)]+=1
 if seen!=Counter({(r,m):1 for r in range(1,REPS+1) for m in MODES}):
  raise ValueError("incomplete or duplicate trials")
 return {(int(r["repeat"]),r["mode"]):float(r["tokens_per_sec"]) for r in rows}
def report(rows):
 obs=validate(rows)
 modes=[]
 for name in MODES:
  scores=[obs[(r,name)] for r in range(1,REPS+1)]
  modes.append(dict(mode=name,median_tokens_per_sec=round(statistics.median(scores),4),
    min_tokens_per_sec=round(min(scores),4),max_tokens_per_sec=round(max(scores),4)))
 paired=[]
 for base in MODES:
  if base=="adaptive_hybrid2":continue
  ratios=[obs[(r,"adaptive_hybrid2")]/obs[(r,base)] for r in range(1,REPS+1)]
  paired.append(dict(baseline=base,median_hybrid_over_baseline=round(statistics.median(ratios),4),
   wins=sum(v>1 for v in ratios),ratios=[round(v,4) for v in ratios]))
 return dict(status="research_only",model="Qwen3.5-0.8B Q4_0",host="VPS02 2 QEMU vCPU SSE2",
    paired_repetitions=REPS,observations=len(rows),modes=modes,paired=paired,
    verdict="DO_NOT_AUTO_PROMOTE",limits="Only 5 repeats, one host, short 24-token decode, noisy shared VPS.")
class Tests(unittest.TestCase):
 def records(self):
  return [dict(repeat=r,mode=m,micro_tasks=t,scheduler=q,hook_hit=h,dispatch_hit=d,
    scheduler_hit=s,worker_threads=2,n_gen=24,bench_repeats=2,
    tokens_per_sec="4.0",stddev_tokens_per_sec="0.1",build_commit="lab")
    for r in range(1,REPS+1) for m,(t,q,h,d,s) in MODES.items()]
 def test_good(self):self.assertEqual(report(self.records())["observations"],25)
 def test_missing(self):
  with self.assertRaises(ValueError):report(self.records()[:-1])
 def test_trace(self):
  a=self.records();a[0]["hook_hit"]=1
  with self.assertRaises(ValueError):report(a)
if __name__=="__main__":
 if len(sys.argv)==2 and sys.argv[1]=="--test":unittest.main(argv=["p35"],exit=False)
 elif len(sys.argv)==2:
  with open(sys.argv[1],newline="") as f: rows=list(csv.DictReader(f))
  print(json.dumps(report(rows),indent=2,sort_keys=True))
 else:sys.exit("usage: validator --test|results.csv")
