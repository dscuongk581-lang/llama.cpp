#!/usr/bin/env python3
"""P3.4 randomized whole-Qwen latency and throughput on single VPS02.
Compare 2-worker native, SSE2 hook, adaptive atomic, adaptive static,
and fixed8 static. Never enables production mode.
"""
import argparse,csv,os,random,statistics,subprocess
from pathlib import Path
MODES=[("native2","0","0","0","0"),("hook2","1","0","0","0"),
       ("adaptive_atomic2","1","1","adaptive","0"),
       ("adaptive_static2","1","1","adaptive","static"),
       ("fixed8_static2","1","1","8","static")]
COLS=["repeat","order","mode","threads","micro_tasks","scheduler","token_limit","bench_repeats",
      "tokens_per_sec","stddev_tokens_per_sec","build_commit","hook_hit","dispatch_hit","static_hit"]
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--binary",type=Path,required=True);p.add_argument("--model",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True);p.add_argument("--repeats",type=int,default=5)
 a=p.parse_args()
 rng=random.Random(340113);rows=[]
 for rep in range(1,a.repeats+1):
  choices=MODES[:];rng.shuffle(choices)
  for order,(mode,hook,micro,tasks,static) in enumerate(choices,1):
   env=os.environ.copy()
   env.update(GGML_VCPU_LAB_ENABLE=hook,GGML_VCPU_MICRO_ENABLE=micro,
       GGML_VCPU_MICRO_TASKS=tasks,GGML_VCPU_MICRO_SCHED=static,
       GGML_VCPU_LAB_TRACE="1",GGML_VCPU_MICRO_TRACE="1",GGML_VCPU_MICRO_STATS="0")
   process=subprocess.run(["nice","-n","15",str(a.binary),"-m",str(a.model),"--offline","-p","0",
       "-n","24","-t","2","-r","2","-ngl","0","-o","csv"],env=env,capture_output=True,
       text=True,timeout=130,check=True)
   entries=list(csv.DictReader(process.stdout.splitlines()))
   if len(entries)!=1:raise ValueError(f"bad CSV {mode}")
   result=entries[0]
   hook_hit="VLANIX_VCPU_LAB_HOOK_HIT" in process.stderr
   dispatch_hit="VLANIX_MICRO_VCPU_DISPATCH" in process.stderr
   static_hit="VLANIX_MICRO_VCPU_SCHED mode=static" in process.stderr
   if (hook_hit,dispatch_hit,static_hit)!=(hook=="1",micro=="1",static=="static"):
    raise ValueError(f"bad route {mode} {(hook_hit,dispatch_hit,static_hit)}")
   if int(result["n_threads"])!=2 or int(result["n_gen"])!=24:raise ValueError("bad benchmark config")
   rec=dict(repeat=rep,order=order,mode=mode,threads=2,micro_tasks=tasks,scheduler=static,
      token_limit=24,bench_repeats=2,tokens_per_sec=float(result["avg_ts"]),
      stddev_tokens_per_sec=float(result["stddev_ts"]),build_commit=result["build_commit"],
      hook_hit=int(hook_hit),dispatch_hit=int(dispatch_hit),static_hit=int(static_hit))
   rows.append(rec)
   with open(a.output,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=COLS);w.writeheader();w.writerows(rows)
   print("P34_PROGRESS",rep,order,mode,round(rec["tokens_per_sec"],4),flush=True)
 for name,*_ in MODES:
  values=[r["tokens_per_sec"] for r in rows if r["mode"]==name]
  print("P34_MEDIAN",name,round(statistics.median(values),4),flush=True)
 for mode in ("native2","hook2","adaptive_atomic2","fixed8_static2"):
  ratios=[next(r["tokens_per_sec"] for r in rows if r["mode"]=="adaptive_static2" and r["repeat"]==i)/
          next(r["tokens_per_sec"] for r in rows if r["mode"]==mode and r["repeat"]==i)
          for i in range(1,a.repeats+1)]
  print("P34_PAIR_STATIC_OVER",mode,round(statistics.median(ratios),4),
      sum(x>1 for x in ratios),[round(x,3) for x in ratios],flush=True)
 print("P34_SWEEP_COMPLETE",len(rows),flush=True)
if __name__=="__main__":main()
