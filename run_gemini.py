"""LLM track runner — evaluate Gemini on all eval languages.

Gemini is API-based (no GPU), so this can run alongside the GPU pipeline.
Writes to benchmarks_llm/ (merged into benchmarks/ later).

Run:  python3 run_gemini.py 2>&1 | tee /tmp/nsanku_gemini.log
      python3 run_gemini.py --langs twi ewe hau
"""
import argparse
import os
import sys
import time

sys.path.insert(0, ".")
if os.path.isdir("/mnt/volume_d2wey28/hf_cache"):
    os.environ.setdefault("HF_HOME", "/mnt/volume_d2wey28/hf_cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/volume_d2wey28/hf_cache/hub")

from benchmark.gemini import evaluate_gemini
from benchmark.evaluate import load_eval_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", help="ISO codes (default: all eval langs)")
    args = ap.parse_args()
    langs = args.langs or list(load_eval_configs().keys())

    print("=" * 60)
    print(f"  nsanku-ASR — LLM track (Gemini) · {len(langs)} languages")
    print("=" * 60)
    t0 = time.time()
    for i, iso in enumerate(langs, 1):
        print(f"\n===== [{i}/{len(langs)}] {iso} =====", flush=True)
        try:
            evaluate_gemini(iso)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
    print(f"\nLLM track done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
