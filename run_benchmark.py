#!/usr/bin/env python3
"""nsanku-ASR: Benchmark org-owned ASR models on Ghanaian languages.

Scores each model per eval category (bible/jw/finance/unicef) and averages
across the categories a language appears in.

Usage:
    python run_benchmark.py                              # All eval languages
    python run_benchmark.py --langs twi ewe dag          # Specific languages
    python run_benchmark.py --langs twi --models whisper # Filter by model name
    python run_benchmark.py --dry-run                     # Preview (no GPU)
"""
import argparse
import json

from benchmark.config import EVAL_CONFIGS_FILE


def eval_languages():
    return list(json.load(open(EVAL_CONFIGS_FILE)).keys())


def dry_run(langs, model_filter=None):
    from benchmark.evaluate import get_language_models, language_categories
    print(f"{'ISO':6s} {'Cats':22s} {'#Mdl':5s}  Models")
    print("-" * 100)
    for iso in langs:
        cats = [c for c, _ in language_categories(iso)]
        models = get_language_models(iso)
        if model_filter:
            models = [m for m in models if model_filter.lower() in m["name"].lower()]
        names = ", ".join(m["name"] for m in models[:4])
        if len(models) > 4:
            names += f" ... +{len(models) - 4}"
        print(f"{iso:6s} {','.join(cats):22s} {len(models):<5d}  {names}")


def main():
    parser = argparse.ArgumentParser(description="nsanku-ASR: org ASR benchmark")
    parser.add_argument("--langs", nargs="+", help="ISO codes (default: all eval langs)")
    parser.add_argument("--models", help="Filter model names containing this string")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="re-run models even if already scored on every current category")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    langs = args.langs or eval_languages()
    print("=" * 60)
    print("  nsanku-ASR — org ASR benchmark runner")
    print("=" * 60)
    print(f"  Languages: {len(langs)}   Model filter: {args.models or 'all'}   Device: {args.device}")

    if args.dry_run:
        print()
        dry_run(langs, args.models)
        return

    from benchmark.evaluate import evaluate_language
    for iso in langs:
        evaluate_language(iso, model_filter=args.models, device=args.device,
                          force=args.force)


if __name__ == "__main__":
    main()
