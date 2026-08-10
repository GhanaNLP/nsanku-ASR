"""Hosted-API track runner — evaluate the Khaya ASR API on all supported eval languages.

API-based (no GPU). Writes to benchmarks_api/ (merged into benchmarks/ later).

Run:  python3 run_khaya.py 2>&1 | tee /tmp/nsanku_khaya.log
      python3 run_khaya.py --langs twi ewe hau
"""
import argparse
import os
import sys
import time

sys.path.insert(0, ".")
os.environ.setdefault("HF_HOME", "/mnt/volume_d2wey28/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/volume_d2wey28/hf_cache/hub")

from benchmark.khaya import evaluate_khaya, EVAL_TO_KHAYA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", help="ISO codes (default: all Khaya-supported eval langs)")
    args = ap.parse_args()
    langs = args.langs or list(EVAL_TO_KHAYA.keys())

    print("=" * 60)
    print(f"  nsanku-ASR - Khaya API track - {len(langs)} languages")
    print("=" * 60)
    t0 = time.time()
    for i, iso in enumerate(langs, 1):
        print(f"\n===== [{i}/{len(langs)}] {iso} =====", flush=True)
        try:
            evaluate_khaya(iso)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
    print(f"\nKhaya track done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
