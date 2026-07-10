#!/usr/bin/env python3
"""Generate a CSV summary of all model evaluation statuses across languages.

Output: model_status.csv with columns:
  model, language, params, wer, cer, status, fail_reason

Usage:
    python3 generate_model_status.py
"""
import csv
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from benchmark.config import RESULTS_FILE, BENCHMARK_DIR

# All languages that were evaluated
ALL_LANGS = [
    "twi","ewe","hau","kbp","dag","dga","fat","nko",
    "any","avn","bud","bim","biv","bib","bwu","ncu",
    "ada","mzw","ffm","acd","gjn","xsm","xon","kma",
    "kus","lef","maw","naw","gur","ntr","nzi","sig",
    "sfw","lip","snw","sil","akp","tpm","kdh","bov","vag",
]

HF_LANGS = ["twi","ewe","hau","kbp","dag","dga","fat","nko"]


def main():
    """Walk every language's YAML, build pass/fail table for all models attempted."""
    rows = []
    seen = set()

    # Load all models we intended to test from ghana_asr_results.json
    with open(RESULTS_FILE) as f:
        results_data = json.load(f)

    for iso in ALL_LANGS:
        bench_path = BENCHMARK_DIR / f"{iso}.yaml"
        if not bench_path.exists():
            continue

        with open(bench_path) as f:
            bench_data = yaml.safe_load(f) or {}

        for b in bench_data.get("benchmarks", []):
            model = b.get("model", "?")
            key = (model, iso)
            if key in seen:
                continue
            seen.add(key)

            wer = b.get("wer")
            error = b.get("error")

            if wer is not None:
                status = "pass"
                reason = ""
            elif error:
                if "gated" in error.lower():
                    status = "fail"
                    reason = "gated_repo"
                elif "arch" in error.lower():
                    status = "fail"
                    reason = error
                elif "load" in error.lower():
                    status = "fail"
                    reason = error
                else:
                    status = "fail"
                    reason = error[:80]
            else:
                status = "fail"
                reason = "unknown"

            rows.append({
                "model": model,
                "language": iso,
                "params": b.get("params", "?"),
                "wer": round(wer, 4) if wer is not None else "",
                "cer": round(b.get("cer", 0), 4) if b.get("cer") is not None else "",
                "status": status,
                "fail_reason": reason,
            })

    # Add models that were never attempted (still in results JSON but not in any YAML)
    for iso in HF_LANGS:
        lang = results_data["languages"].get(iso, {})
        for m in lang.get("asr_models", []):
            name = m["name"]
            if any(kw in name.lower() for kw in ["tts", "text-to-speech"]):
                continue
            key = (name, iso)
            if key not in seen:
                seen.add(key)
                rows.append({
                    "model": name,
                    "language": iso,
                    "params": m.get("size", "?"),
                    "wer": "",
                    "cer": "",
                    "status": "not_attempted",
                    "fail_reason": "",
                })

    # Save CSV
    out_path = ROOT / "model_status.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "language", "params", "wer", "cer", "status", "fail_reason"])
        w.writeheader()
        w.writerows(rows)

    # Summary
    passed = sum(1 for r in rows if r["status"] == "pass")
    failed = sum(1 for r in rows if r["status"] == "fail")
    not_attempted = sum(1 for r in rows if r["status"] == "not_attempted")
    print(f"model_status.csv: {len(rows)} entries")
    print(f"  Pass: {passed}")
    print(f"  Fail: {failed}")
    print(f"  Not attempted: {not_attempted}")

    # Show fail reasons
    reasons = {}
    for r in rows:
        if r["status"] == "fail":
            reason = r["fail_reason"][:40]
            reasons[reason] = reasons.get(reason, 0) + 1
    if reasons:
        print("\n  Fail reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {count:3d}  {reason}")


if __name__ == "__main__":
    main()