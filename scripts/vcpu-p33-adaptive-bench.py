#!/usr/bin/env python3
"""VLANIX P3.3 actual Qwen decode benchmark, randomized by repeat, 5 profiles.
Native, SSE2, fixed8 vs adaptive (2 workers), and adaptive 1 worker.
No GGUF weights or generated text written. Only aggregate per-run metrics.
"""
import argparse,csv,os,random,subprocess,statistics
from pathlib import Path
MODES=[("native2",2,False,"0"),("hook2",2,True,"0"),("fixed8_2",2,True,"8"),("adaptive2",2,True,"adaptive"),("adaptive1",1,True,"adaptive")]
HEAD=["repeat","order","mode","workers","logical","tokens","avg_ts","stddev_ts","commit","hook_hit","dispatch_hit"]
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--binary",type=Path,required=True);p.add_argument("--model",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True);p.add_argument("--repeats",type=int,default=5)
 a=p.parse_args()
 if a.repeats<1 or a.repeats>20:p.error("bad repeat")
 rng=random.Random(330113)
 entries=[]
 for rep in range(1,a.repeats+1):
  modes=MODES[:];rng.shuffle(modes)
  for order,(mode,workers,hook,logical) in enumerate(modes,1):
   env=os.environ.copy();env.update(GGML_VCPU_LAB_ENABLE="1" if hook else "0",GGML_VCPU_MICRO_ENABLE="1" if logical!="0" else "0",GGML_VCPU_MICRO_TASKS=logical,
    GGML_VCPU_LAB_TRACE="1",GGML_VCPU_MICRO_TRACE="1",GGML_VCPU_MICRO_STATS="0")
   cmd=["nice","-n","15",str(a.binary),"-m",str(a.model),"--offline","-p","0","-n","24","-t",str(workers),"-r","2","-ngl","0","-o","csv"]
   run=subprocess.run(cmd,env=env,text=True,capture_output=True,check=True,timeout=130)
   parsed=list(csv.DictReader(run.stdout.splitlines()))
   if len(parsed)!=1:raise ValueError("bad csv "+mode)
   x=parsed[0]
   hit="VLANIX_VCPU_LAB_HOOK_HIT" in run.stderr
   expected_trace= "VLANIX_MICRO_VCPU_DISPATCH mode=adaptive" if logical=="adaptive" else "VLANIX_MICRO_VCPU_DISPATCH units=8" if logical=="8" else ""
   dispatch=bool(expected_trace and expected_trace in run.stderr)
   if hit!=hook or dispatch!=(logical!="0"):raise ValueError("hook/dispatch mismatch "+mode+" "+run.stderr[-200:])
   if int(x["n_threads"])!=workers or int(x["n_gen"])!=24:raise ValueError("invalid run config")
   row={"repeat":rep,"order":order,"mode":mode,"workers":workers,"logical":logical,"tokens":24,
     "avg_ts":float(x["avg_ts"]),"stddev_ts":float(x["stddev_ts"]),"commit":x["build_commit"],
     "hook_hit":int(hit),"dispatch_hit":int(dispatch)}
   entries.append(row)
   with open(a.output,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=HEAD);w.writeheader();w.writerows(entries)
   print("P33_RUN",rep,order,mode,f'{row["avg_ts"]:.4f}',flush=True)
 for name,*_ in MODES:
  arr=[x["avg_ts"] for x in entries if x["mode"]==name]
  print("P33_MEDIAN",name,f'{statistics.median(arr):.4f}',flush=True)
 for mode in ("adaptive2","fixed8_2","hook2"):
  ratios=[next(x["avg_ts"] for x in entries if x["repeat"]==i and x["mode"]=="adaptive2")/
          next(x["avg_ts"] for x in entries if x["repeat"]==i and x["mode"]==mode) for i in range(1,a.repeats+1)]
  print("P33_ADAPTIVE2_OVER",mode,f'{statistics.median(ratios):.4f}',[round(x,3) for x in ratios],flush=True)
 print("P33_COMPLETE",len(entries),flush=True)
if __name__=="__main__":main()
