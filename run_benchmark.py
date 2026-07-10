#!/usr/bin/env python3
"""nsanku-ASR: Run HF ASR model benchmarks on Ghanaian languages.

Usage:
    python run_benchmark.py                              # All languages with models
    python run_benchmark.py --langs twi ewe dag           # Specific languages
    python run_benchmark.py --langs twi --models whisper  # Filter by model name
    python run_benchmark.py --dry-run                     # Preview
"""
import argparse
import json
from benchmark.config import RESULTS_FILE, LANG_TO_CONFIG


def get_languages_with_models():
    with open(RESULTS_FILE) as f:
        return list(json.load(f)["languages"].keys())


def count_models(iso_code):
    with open(RESULTS_FILE) as f:
        return len(json.load(f)["languages"].get(iso_code, {}).get("asr_models", []))


def get_unique_models_for_lang(iso_code):
    """Return unique model IDs for a language."""
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    models = data["languages"].get(iso_code, {}).get("asr_models", [])
    # Deduplicate by name
    seen = set()
    unique = []
    for m in models:
        if m["name"] not in seen:
            seen.add(m["name"])
            unique.append(m)
    return unique


def dry_run(langs, model_filter=None):
    print(f"{'ISO':6s} {'Models':6s}  {'Models list':55s}   {'Dataset':30s}")
    print(f"{'-' * 100}")
    for iso in langs:
        models = get_unique_models_for_lang(iso)
        if model_filter:
            models = [m for m in models if model_filter.lower() in m["name"].lower()]
        names = ", ".join(m["name"] for m in models[:6])
        if len(models) > 6:
            names += f" ... +{len(models)-6} more"
        config = LANG_TO_CONFIG.get(iso, "?")
        print(f"{iso:6s} {str(len(models)):6s}  {names:55s}   {config:30s}")


def main():
    parser = argparse.ArgumentParser(description="nsanku-ASR: HF model benchmarks")
    parser.add_argument("--langs", nargs="+", help="ISO codes (default: all)")
    parser.add_argument("--models", help="Filter model names containing this string")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    langs = args.langs or get_languages_with_models()

    print("=" * 60)
    print("  nsanku-ASR — HF ASR Benchmark Runner")
    print("=" * 60)
    print(f"  Languages: {len(langs)}")
    print(f"  Model filter: {args.models or 'all'}")
    print(f"  Device: {args.device}")

    if args.dry_run:
        print()
        dry_run(langs, args.models)
        return

    from benchmark.evaluate import evaluate_language

    for iso in langs:
        evaluate_language(iso, model_filter=args.models, device=args.device)


if __name__ == "__main__":
    main()
