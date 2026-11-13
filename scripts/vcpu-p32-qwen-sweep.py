#!/usr/bin/env python3
"""P3.2 randomized single-VPS Qwen decode: 8 logical tasks on 1/2 workers.
GGML native/previous hook baseline included. Runtime/library all same fork build.
Outputs performance metadata, not text or model weights.
"""
import argparse
import csv
import os
import random
import statistics
import subprocess
from pathlib import Path

PROFILES = [
    ("native2",2,"0","0"),
    ("micro8_worker1",1,"1","1"),
    ("micro8_worker2",2,"1","1"),
    ("hook2_no_micro",2,"1","0"),
]
COLUMNS = ["repeat","order","mode","worker_threads","micro_tasks","commit",
           "n_gen","n_prompt","bench_repeats","tokens_per_sec","std_tokens_per_sec",
           "avg_ns","hook_hit","micro_dispatch_hit"]

def run(exe,model,raw,repetitions,tokens):
    rng=random.Random(320113)
    result=[]
    for rep in range(1,repetitions+1):
        batch=PROFILES[:]
        rng.shuffle(batch)
        for order,(name,threads,hook,micro) in enumerate(batch,1):
            env=os.environ.copy()
            env.update(GGML_VCPU_LAB_ENABLE=hook,GGML_VCPU_MICRO_ENABLE=micro,
                       GGML_VCPU_MICRO_TASKS="8",GGML_VCPU_MICRO_TRACE="1",
                       GGML_VCPU_LAB_TRACE="1",GGML_VCPU_MICRO_STATS="0")
            proc=subprocess.run(["nice","-n","15",str(exe),"-m",str(model),
                    "--offline","-p","0","-n",str(tokens),"-t",str(threads),
                    "-r","2","-ngl","0","-o","csv"],
                    text=True,capture_output=True,check=True,timeout=130,env=env)
            items=list(csv.DictReader(proc.stdout.splitlines()))
            if len(items)!=1:raise RuntimeError(f"unexpected benchmark records: {name} {proc.stdout[:160]}")
            value=items[0]
            hook_hit="VLANIX_VCPU_LAB_HOOK_HIT" in proc.stderr
            micro_hit="VLANIX_MICRO_VCPU_DISPATCH units=8 workers="+str(threads) in proc.stderr
            if hook_hit !=(hook=="1") or micro_hit !=(micro=="1"):
                raise RuntimeError(f"unexpected hook or micro dispatch in {name}: hook={hook_hit} micro={micro_hit}")
            if int(value["n_threads"])!=threads or int(value["n_gen"])!=tokens or int(value["n_prompt"])!=0:
                raise RuntimeError("wrong benchmark parameters")
            record=dict(repeat=rep,order=order,mode=name,worker_threads=threads,
                        micro_tasks=(8 if micro=="1" else 0),
                        commit=value["build_commit"],n_gen=tokens,n_prompt=0,bench_repeats=2,
                        tokens_per_sec=float(value["avg_ts"]),
                        std_tokens_per_sec=float(value["stddev_ts"]),avg_ns=value["avg_ns"],
                        hook_hit=int(hook_hit),micro_dispatch_hit=int(micro_hit))
            if record["tokens_per_sec"]<=0: raise RuntimeError("invalid token/s")
            result.append(record)
            # Append after each run: preserve partial evidence on process interruption.
            with open(raw,"w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=COLUMNS)
                writer.writeheader();writer.writerows(result)
            print(f"P32_PROGRESS {rep}/{repetitions} {name} {record['tokens_per_sec']:.4f} tok/s",flush=True)
    grouped={m:[r["tokens_per_sec"] for r in result if r["mode"]==m] for m,*_ in PROFILES}
    for m,vals in grouped.items():
        print(f"P32_MODE {m} median={statistics.median(vals):.4f} min={min(vals):.4f} max={max(vals):.4f}",flush=True)
    paired=[]
    for rep in range(1,repetitions+1):
        a=next(r["tokens_per_sec"] for r in result if r["repeat"]==rep and r["mode"]=="micro8_worker2")
        b=next(r["tokens_per_sec"] for r in result if r["repeat"]==rep and r["mode"]=="native2")
        paired.append(a/b)
    print(f"P32_PAIRED_MICRO2_VS_NATIVE2 median={statistics.median(paired):.4f} ratios={[round(x,3) for x in paired]}",flush=True)
    print("P32_BENCH_COMPLETE records="+str(len(result)),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--exe",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--repetitions",type=int,default=5)
    p.add_argument("--tokens",type=int,default=24)
    a=p.parse_args()
    if a.repetitions<1 or a.repetitions>32 or a.tokens<1: p.error("invalid sampling")
    run(a.exe,a.model,a.output,a.repetitions,a.tokens)
