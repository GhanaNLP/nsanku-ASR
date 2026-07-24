"""Master pipeline: benchmark org-owned ASR models across all eval languages.

Scoring is per-category (bible / jw / finance / unicef), averaged across the
categories each language appears in. Gemini is not part of this benchmark.

Run:  python3 pipeline.py 2>&1 | tee /tmp/nsanku_pipeline.log
"""
import os, sys, time, gc, json

import torch
sys.path.insert(0, ".")

os.environ.setdefault("HF_HOME", "/mnt/volume_d2wey28/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/volume_d2wey28/hf_cache/hub")

from benchmark.config import EVAL_CONFIGS_FILE
from benchmark.evaluate import evaluate_language


def eval_languages():
    return list(json.load(open(EVAL_CONFIGS_FILE)).keys())


def main():
    langs = eval_languages()
    print("=" * 60, flush=True)
    print(f"  nsanku-ASR — org-only ASR benchmark ({len(langs)} languages)", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    for i, iso in enumerate(langs, 1):
        print(f"\n===== [{i}/{len(langs)}] {iso} =====", flush=True)
        try:
            evaluate_language(iso)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nAll done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
