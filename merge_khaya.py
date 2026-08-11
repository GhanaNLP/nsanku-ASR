"""Merge the hosted-API track (benchmarks_api/) into benchmarks/.

Adds the Khaya ASR API entry into benchmarks/{iso}.yaml (creating the file if
the language had no other models), tags every entry with model_class, and
re-ranks by WER. Idempotent.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, ".")
from benchmark.config import BENCHMARK_DIR, NUM_SAMPLES
from benchmark.evaluate import load_eval_configs

API_DIR = Path("benchmarks_api")


def _rank(benchmarks):
    return sorted(benchmarks, key=lambda x: (x.get("wer") is None, x.get("wer") or 1e9))


def main():
    cfg = load_eval_configs()
    merged = 0
    for api_path in sorted(API_DIR.glob("*.yaml")):
        iso = api_path.stem
        api = yaml.safe_load(open(api_path)) or {}
        api_entries = api.get("benchmarks", [])
        for e in api_entries:
            e.setdefault("model_class", "non-llm")

        out_path = BENCHMARK_DIR / f"{iso}.yaml"
        if out_path.exists():
            base = yaml.safe_load(open(out_path)) or {}
        else:
            meta = cfg.get(iso, {})
            base = {
                "iso_639_3": iso,
                "language": api.get("language") or meta.get("language", iso),
                "num_samples_per_category": NUM_SAMPLES,
                "categories": api.get("categories") or [c["category"] for c in meta.get("categories", [])],
                "benchmarks": [],
            }

        by_model = {b["model"]: b for b in base.get("benchmarks", [])}
        for b in by_model.values():
            b.setdefault("model_class", "non-llm")
        for e in api_entries:
            by_model[e["model"]] = e

        base["benchmarks"] = _rank(list(by_model.values()))
        with open(out_path, "w") as f:
            yaml.dump(base, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        merged += 1
        print(f"  merged {iso}: +{len(api_entries)} api entry -> {len(base['benchmarks'])} total")
    print(f"\nMerged API results into {merged} languages.")


if __name__ == "__main__":
    main()
