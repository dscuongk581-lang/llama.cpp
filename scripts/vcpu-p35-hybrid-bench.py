#!/usr/bin/env python3
"""VLANIX P3.5 Qwen SSE2 local hybrid vs atomic and static, paired 5x5 randomized.
Lab-only. No model weights/prompts emitted.
"""
import argparse,csv,os,random,subprocess,statistics
from pathlib import Path
MODES=[
 ("native2","0","0","0","0"),
 ("hook2","1","0","0","0"),
 ("adaptive_atomic2","1","1","adaptive","0"),
 ("adaptive_static2","1","1","adaptive","static"),
 ("adaptive_hybrid2","1","1","adaptive","hybrid")]
COLS=["repeat","order","mode","worker_threads","micro_tasks","scheduler",
      "n_gen","bench_repeats","tokens_per_sec","stddev_tokens_per_sec",
      "build_commit","hook_hit","dispatch_hit","scheduler_hit"]
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--exe",type=Path,required=True)
 p.add_argument("--model",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 p.add_argument("--repeats",type=int,default=5)
 a=p.parse_args();rng=random.Random(350113);rows=[]
 for rep in range(1,a.repeats+1):
  choices=MODES[:];rng.shuffle(choices)
  for order,(name,hook,micro,tasks,scheduler) in enumerate(choices,1):
   env=os.environ.copy()
   env.update(GGML_VCPU_LAB_ENABLE=hook,GGML_VCPU_MICRO_ENABLE=micro,
     GGML_VCPU_MICRO_TASKS=tasks,GGML_VCPU_MICRO_SCHED=scheduler,
     GGML_VCPU_LAB_TRACE="1",GGML_VCPU_MICRO_TRACE="1",GGML_VCPU_MICRO_STATS="0")
   cmd=["nice","-n","15",str(a.exe),"-m",str(a.model),"--offline",
     "-p","0","-n","24","-t","2","-r","2","-ngl","0","-o","csv"]
   out=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=140,check=True)
   data=list(csv.DictReader(out.stdout.splitlines()))
   if len(data)!=1:raise ValueError(f"invalid CSV {name}")
   d=data[0]
   hit="VLANIX_VCPU_LAB_HOOK_HIT" in out.stderr
   dispatch="VLANIX_MICRO_VCPU_DISPATCH mode=adaptive" in out.stderr
   sch=bool(scheduler in ("static","hybrid") and
      ("VLANIX_MICRO_VCPU_SCHED mode="+scheduler) in out.stderr)
   if (hit,dispatch,sch)!=(hook=="1",micro=="1",scheduler in ("static","hybrid")):
    raise ValueError(f"incorrect dispatch {name} {hit} {dispatch} {sch}")
   if int(d["n_threads"])!=2 or int(d["n_gen"])!=24:raise ValueError("wrong bench parameters")
   r=dict(repeat=rep,order=order,mode=name,worker_threads=2,
      micro_tasks=tasks,scheduler=scheduler,n_gen=24,bench_repeats=2,
      tokens_per_sec=float(d["avg_ts"]),stddev_tokens_per_sec=float(d["stddev_ts"]),
      build_commit=d["build_commit"],hook_hit=int(hit),
      dispatch_hit=int(dispatch),scheduler_hit=int(sch))
   rows.append(r)
   with open(a.output,"w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=COLS);writer.writeheader();writer.writerows(rows)
   print("P35_RUN",rep,order,name,f'{r["tokens_per_sec"]:.4f}',flush=True)
 for name,*_ in MODES:
  vals=[x["tokens_per_sec"] for x in rows if x["mode"]==name]
  print("P35_MEDIAN",name,f'{statistics.median(vals):.4f}',flush=True)
 for name,*_ in MODES:
  if name=="adaptive_hybrid2":continue
  ratios=[next(x["tokens_per_sec"] for x in rows if x["repeat"]==rep and x["mode"]=="adaptive_hybrid2")/
      next(x["tokens_per_sec"] for x in rows if x["repeat"]==rep and x["mode"]==name) for rep in range(1,a.repeats+1)]
  print("P35_PAIRED_HYBRID_OVER",name,f'{statistics.median(ratios):.4f}',
        sum(v>1 for v in ratios),[round(v,3) for v in ratios],flush=True)
 print("P35_COMPLETE",len(rows),flush=True)
if __name__=="__main__":main()
