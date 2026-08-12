"""Google Speech Recognition track runner.

API-based (no GPU). Writes to benchmarks_api/ (merged into benchmarks/ later).

Run:  python3 run_google.py --langs twi ewe gaa
      python3 run_google.py --langs twi --samples 20   # quick pilot
"""
import argparse
import os
import sys
import time

sys.path.insert(0, ".")

from benchmark.google import evaluate_google, EVAL_TO_GOOGLE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", help="ISO codes (default: all Google-supported eval langs)")
    ap.add_argument("--samples", type=int, default=1000, help="samples per category")
    args = ap.parse_args()
    langs = args.langs or list(EVAL_TO_GOOGLE.keys())

    print("=" * 60)
    print(f"  nsanku-ASR - Google ASR track - {len(langs)} languages "
          f"({args.samples} samples/category)")
    print("=" * 60)
    t0 = time.time()
    for i, iso in enumerate(langs, 1):
        print(f"\n===== [{i}/{len(langs)}] {iso} =====", flush=True)
        try:
            evaluate_google(iso, num_samples=args.samples)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
    print(f"\nGoogle track done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
