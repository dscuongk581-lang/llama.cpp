#!/usr/bin/env python3
"""P3.4 rigorous shape/mode/config integrity and paired inference evaluation.
NO automatic promotion on small N.
"""
import csv,json,math,statistics,sys,unittest
from collections import Counter
PROFILES={"native2":("0","0","0",0,0,0),
"hook2":("0","0","0",1,0,0),
"adaptive_atomic2":("adaptive","0","0",1,1,0),
"adaptive_static2":("adaptive","static","0",1,1,1),
"fixed8_static2":("8","static","0",1,1,1)}
REPS=5
def validate(rows):
 counts=Counter()
 for r in rows:
  m=r["mode"]
  if m not in PROFILES:raise ValueError("unexpected profile")
  tasks,sched,_,h,d,st=PROFILES[m]
  if (r["micro_tasks"],r["scheduler"],int(r["hook_hit"]),int(r["dispatch_hit"]),int(r["static_hit"])) != (tasks,sched,h,d,st):
   raise ValueError("route mismatch")
  if (int(r["threads"]),int(r["token_limit"]),int(r["bench_repeats"]))!=(2,24,2):raise ValueError("wrong run parameters")
  if not r["build_commit"]:raise ValueError("missing build")
  for name in ("tokens_per_sec","stddev_tokens_per_sec"):
   if not math.isfinite(float(r[name])):raise ValueError("nonfinite")
  if float(r["tokens_per_sec"])<=0 or float(r["stddev_tokens_per_sec"])<0:raise ValueError("bad timing")
  rep=int(r["repeat"])
  if not 1<=rep<=REPS:raise ValueError("repeat out of bounds")
  counts[(rep,m)]+=1
 if counts!=Counter({(rep,m):1 for rep in range(1,REPS+1) for m in PROFILES}):
  raise ValueError("incomplete or duplicate randomized sweep")
 return {(int(r["repeat"]),r["mode"]):float(r["tokens_per_sec"]) for r in rows}
def report(rows):
 x=validate(rows);med=[]
 for m in PROFILES:
  values=[x[(i,m)] for i in range(1,REPS+1)]
  med.append(dict(mode=m,median_tokens_per_sec=round(statistics.median(values),4),
                  minimum_tokens_per_sec=round(min(values),4),maximum_tokens_per_sec=round(max(values),4)))
 paired=[]
 for m in ("native2","hook2","adaptive_atomic2","fixed8_static2"):
  ratio=[x[(i,"adaptive_static2")]/x[(i,m)] for i in range(1,REPS+1)]
  paired.append(dict(baseline=m,median_static_over_baseline=round(statistics.median(ratio),4),
       static_faster_pairs=sum(a>1 for a in ratio),per_repeat=[round(a,4) for a in ratio]))
 return dict(status="offline_lab_only",source="VPS02 SSE2 QEMU shared 2 vCPU",
     repetitions=REPS,observations=len(rows),model="Qwen3.5-0.8B Q4_0",
     modes=med,paired=paired,
     deployment_decision="NO_AUTOMATIC_PROMOTION",
     warning="Small noisy sample. Actual logits checked separately. Single-host only.")
class UnitTests(unittest.TestCase):
 def make(self):
  rows=[]
  for rep in range(1,REPS+1):
   for m,(tasks,sched,_,h,d,st) in PROFILES.items():
    rows.append(dict(repeat=rep,mode=m,micro_tasks=tasks,scheduler=sched,
       hook_hit=h,dispatch_hit=d,static_hit=st,threads=2,token_limit=24,
       bench_repeats=2,build_commit="test",tokens_per_sec="4",stddev_tokens_per_sec="0.01"))
  return rows
 def test_complete(self):
  self.assertEqual(report(self.make())["observations"],25)
 def test_missing(self):
  with self.assertRaises(ValueError):report(self.make()[:-1])
 def test_wrong_static_route(self):
  r=self.make();r[0]["static_hit"]=1
  with self.assertRaises(ValueError):report(r)
if __name__=="__main__":
 if len(sys.argv)==2 and sys.argv[1]=="--test":
  unittest.main(argv=["p34"],exit=False)
 elif len(sys.argv)==2:
  with open(sys.argv[1],newline="") as f: rows=list(csv.DictReader(f))
  print(json.dumps(report(rows),indent=2,sort_keys=True))
 else:sys.exit("usage: script --test|CSV")
