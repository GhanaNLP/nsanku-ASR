"""LLM track runner — evaluate a hosted Gemini/Gemma model on all eval languages.

These models are API-based (no GPU), so this can run alongside the GPU pipeline.
Writes to benchmarks_llm/ (merged into benchmarks/ later).

One model can be evaluated in several flavours (e.g. thinking on vs off); pass
--thinking/--label to record each as its own model id (google/{model}-{label}).

Run:  python3 run_gemini.py 2>&1 | tee /tmp/nsanku_gemini.log
      python3 run_gemini.py --langs twi ewe hau
      python3 run_gemini.py --model gemma-4-12b-it --thinking high --label thinking
      python3 run_gemini.py --model gemma-4-12b-it --thinking minimal --label nothink
"""
import argparse
import os
import sys
import time

sys.path.insert(0, ".")
if os.path.isdir("/mnt/volume_d2wey28/hf_cache"):
    os.environ.setdefault("HF_HOME", "/mnt/volume_d2wey28/hf_cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/volume_d2wey28/hf_cache/hub")

from benchmark.gemini import evaluate_gemini, GEMINI_MODEL
from benchmark.evaluate import load_eval_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", help="ISO codes (default: all eval langs)")
    ap.add_argument("--model", help="Gemini API model (default: $GEMINI_MODEL)")
    ap.add_argument("--thinking", choices=["high", "minimal"],
                    help="internal reasoning: high = on, minimal = off")
    ap.add_argument("--label", help="suffix for the recorded model id, e.g. nothink")
    ap.add_argument("--max-workers", type=int, help="concurrent requests")
    args = ap.parse_args()
    langs = args.langs or list(load_eval_configs().keys())
    model = args.model or GEMINI_MODEL
    flavour = f"{model}-{args.label}" if args.label else model

    print("=" * 60)
    print(f"  nsanku-ASR — LLM track ({flavour}) · {len(langs)} languages")
    print("=" * 60)
    t0 = time.time()
    for i, iso in enumerate(langs, 1):
        print(f"\n===== [{i}/{len(langs)}] {iso} =====", flush=True)
        try:
            evaluate_gemini(iso, model=args.model, thinking_level=args.thinking,
                            label=args.label, max_workers=args.max_workers)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
    print(f"\nLLM track done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
