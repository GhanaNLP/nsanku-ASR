"""Merge the LLM track (benchmarks_llm/) into benchmarks/.

For each language, adds the Gemini entry into benchmarks/{iso}.yaml (creating the
file if the language had no non-LLM models), tags every entry with model_class,
and re-ranks by WER. Idempotent.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, ".")
from benchmark.config import BENCHMARK_DIR, NUM_SAMPLES
from benchmark.evaluate import load_eval_configs

LLM_DIR = Path("benchmarks_llm")


def _rank(benchmarks):
    return sorted(benchmarks, key=lambda x: (x.get("wer") is None, x.get("wer") or 1e9))


def main():
    cfg = load_eval_configs()
    merged = 0
    for llm_path in sorted(LLM_DIR.glob("*.yaml")):
        iso = llm_path.stem
        llm = yaml.safe_load(open(llm_path)) or {}
        llm_entries = llm.get("benchmarks", [])
        for e in llm_entries:
            e.setdefault("model_class", "llm")

        out_path = BENCHMARK_DIR / f"{iso}.yaml"
        if out_path.exists():
            base = yaml.safe_load(open(out_path)) or {}
        else:
            meta = cfg.get(iso, {})
            base = {
                "iso_639_3": iso,
                "language": llm.get("language") or meta.get("language", iso),
                "num_samples_per_category": NUM_SAMPLES,
                "categories": llm.get("categories") or [c["category"] for c in meta.get("categories", [])],
                "benchmarks": [],
            }

        by_model = {b["model"]: b for b in base.get("benchmarks", [])}
        for b in by_model.values():
            b.setdefault("model_class", "non-llm")
        for e in llm_entries:
            by_model[e["model"]] = e  # llm overrides/added

        base["benchmarks"] = _rank(list(by_model.values()))
        with open(out_path, "w") as f:
            yaml.dump(base, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        merged += 1
        print(f"  merged {iso}: +{len(llm_entries)} llm entries -> {len(base['benchmarks'])} total")
    print(f"\nMerged LLM results into {merged} languages.")


if __name__ == "__main__":
    main()
