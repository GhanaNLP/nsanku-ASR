#!/usr/bin/env python3
"""Print a summary table of all benchmark results."""
import json
import yaml
from pathlib import Path

from benchmark.config import BENCHMARK_DIR, RESULTS_FILE

# Load which languages have models
with open(RESULTS_FILE) as f:
    search = json.load(f)["languages"]

print(f"{'Language':25s} {'ISO':5s} {'Utterances':>10s} {'Hours':>8s} {'Models':>6s}  {'Best WER':>8s} {'Best CER':>8s} {'Best Model'}")
print(f"{'-'*95}")

benchmark_files = sorted(BENCHMARK_DIR.glob("*.yaml"))

for iso, info in sorted(search.items()):
    name = info["language"]
    utts = info["utterances"]
    hours = info["hours"]
    n_models = info["total_asr_models"]

    bench_file = BENCHMARK_DIR / f"{iso}.yaml"
    if bench_file.exists():
        with open(bench_file) as f:
            bench = yaml.safe_load(f) or {}
        entries = [b for b in bench.get("benchmarks", []) if b.get("wer") is not None]
        best = min(entries, key=lambda x: x["wer"]) if entries else None
        if best:
            best_wer = f"{best['wer']:.2%}"
            best_cer = f"{best['cer']:.2%}"
            best_model = best["model"].split("/")[-1][:25]
        else:
            best_wer = "—"
            best_cer = "—"
            best_model = "—"
    else:
        best_wer = "—"
        best_cer = "—"
        best_model = "—"

    print(f"{name:25s} {iso:5s} {str(utts):>10s} {f'{hours:.1f}h':>8s} {str(n_models):>6s}  {best_wer:>8s} {best_cer:>8s} {best_model}")
