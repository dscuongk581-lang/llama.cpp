#!/usr/bin/env python3
"""VLANIX P3.1 deterministic text-output parity on the SAME lab llama-cli binary.

Read-only GGUF, opt-in hook never replaces production binary. Native is always
the default. Compare content between [Start thinking] and CLI timing footer,
not stdout timing stats; record only digests and metadata, not generated text.
This is NOT a logits or exact token-ID equivalence proof.
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import unittest
from pathlib import Path

PROMPTS = {
    "arithmetic": "Continue this sequence with the next five integers: 2, 4, 6, 8,",
    "code": "Write a Python function that returns the factorial of a nonnegative integer.",
    "facts": "Explain in two sentences why the sky appears blue at noon.",
}
PERF_FOOTER = re.compile(r"(?m)^\[ Prompt: .*\| Generation: .*\]\s*$")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

def canonical_generation(output):
    raw = ANSI.sub("", output).replace("\r\n", "\n")
    markers = ("\n[Start thinking]\n", "\n[Start answering]\n")
    indices = [raw.find(mark) for mark in markers if raw.find(mark) >= 0]
    if not indices:
        raise ValueError("CLI output lacks a generation section; cannot compare safely")
    offset = min(indices)
    raw = raw[offset + 1:]
    match = PERF_FOOTER.search(raw)
    if not match:
        raise ValueError("CLI performance footer not found; refuse heuristic comparison")
    payload = raw[:match.start()]
    return payload.rstrip("\n")  # remove footer separator, retain actual content

def test_pair(cli, model, name, prompt, seed, tokens, timeout):
    records = []
    for route in ("native", "vcpu-hook"):
        env = os.environ.copy()
        env["GGML_VCPU_LAB_ENABLE"] = "1" if route == "vcpu-hook" else "0"
        env["GGML_VCPU_LAB_TRACE"] = "1"
        cmd = ["nice", "-n", "15", str(cli), "-m", str(model), "--offline",
               "-p", prompt, "--temp", "0", "--seed", str(seed), "-n", str(tokens),
               "-t", "2", "-ngl", "0", "--simple-io", "--single-turn",
               "--no-display-prompt", "--no-warmup", "--log-disable"]
        result = subprocess.run(cmd, input="", text=True, capture_output=True,
                                timeout=timeout, env=env, check=True)
        hit = "VLANIX_VCPU_LAB_HOOK_HIT" in result.stderr
        if hit != (route == "vcpu-hook"):
            raise RuntimeError(f"{name}/{route}: unexpected real GGML hook trace {hit}")
        content = canonical_generation(result.stdout)
        record = {"case": name, "route": route, "seed": seed,
                  "predicted_tokens_limit": tokens, "text_chars": len(content),
                  "generation_sha256": hashlib.sha256(content.encode()).hexdigest(),
                  "hook_hit": hit}
        records.append((record, content))
    native, hooked = records
    equal = native[1] == hooked[1]
    return {"case": name, "exact_generated_text_match": equal,
            "native": native[0], "hook": hooked[0]}

class ParserUnitTests(unittest.TestCase):
    def test_ignores_only_timing_footer(self):
        a = "Intro\n[Start thinking]\nhello\nworld\n\n[ Prompt: 2.0 t/s | Generation: 3.0 t/s ]\n\nExiting...\n"
        b = a.replace("2.0", "7.0").replace("3.0", "8.0")
        self.assertEqual(canonical_generation(a), canonical_generation(b))
    def test_catches_generation_changes(self):
        a = "\n[Start thinking]\nhello\n[ Prompt: 2 t/s | Generation: 3 t/s ]\n"
        b = "\n[Start thinking]\nhallo\n[ Prompt: 2 t/s | Generation: 3 t/s ]\n"
        self.assertNotEqual(canonical_generation(a), canonical_generation(b))
    def test_missing_markers_fails_closed(self):
        with self.assertRaises(ValueError):
            canonical_generation("text\nExiting...")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cli", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--test", action="store_true")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--timeout", type=int, default=110)
    a=p.parse_args()
    if a.test:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(ParserUnitTests)
        result=unittest.TextTestRunner().run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    if not a.cli or not a.model or not a.output:
        p.error("--cli --model --output are required")
    cases=[]
    for index,(name,prompt) in enumerate(PROMPTS.items()):
        # Per-prompt seed, same for both routes; deterministic temperature=0.
        case=test_pair(a.cli,a.model,name,prompt,20261113+index,a.tokens,a.timeout)
        cases.append(case)
        print("CASE",name,"MATCH" if case["exact_generated_text_match"] else "MISMATCH",
              case["native"]["generation_sha256"][:16],
              case["hook"]["generation_sha256"][:16],flush=True)
    payload={"status":"real_lab_llama_cli_text_parity_not_logit_parity",
             "model":"Qwen3.5-0.8B-Q4_0", "threads":2,"temperature":0,
             "predicted_tokens_limit":a.tokens,
             "cases":cases,"all_exact_generated_text_match":all(x["exact_generated_text_match"] for x in cases)}
    a.output.write_text(json.dumps(payload,indent=2)+"\n")
    if not payload["all_exact_generated_text_match"]:
        sys.exit(5)

if __name__=="__main__":main()
