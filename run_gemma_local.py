"""LLM track runner — Gemma 4 12B on the local GPU, across the eval languages.

Three variants are recorded as three distinct leaderboard entries, all on the
same bf16 transformers path (see benchmark/gemma_local.py for why Gemma cannot
be run through the Gemini API at all):

  google/gemma-4-12B-it-thinking   base 12B, thinking ON
  google/gemma-4-12B-it-nothink    base 12B, thinking OFF
  yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1   a 12B-it fine-tune

Each variant loads its model ONCE and then walks every language, because the
load is the expensive part (~24GB of weights) and a per-language reload would
dominate the run. Results go to benchmarks_llm/{iso}.yaml, checkpointed per
category, so an interrupted run resumes where it stopped.

COST, measured on the H200 (shared with other jobs):
  * thinking OFF — ~28s per clip, so ~1.5h per 200-clip category.
  * thinking ON  — the model emits ~1024 thought tokens against ~30 for a bare
    transcription, and did not finish a single clip in 20 minutes of wall time.
    At 256 tokens it never closes the thought block at all, so the response has
    no content field and the clip is scored as unanswered. `base-thinking` is
    therefore NOT part of a default run; ask for it explicitly and expect hours
    per category.
A full 45-language sweep is ~1767h (74 days) PER variant even with thinking off,
so pass --langs and NSANKU_NUM_SAMPLES to bound what is actually run.

Run:  python3 run_gemma_local.py 2>&1 | tee gemma_local.log
      python3 run_gemma_local.py --langs twi_asante twi_akuapem
      python3 run_gemma_local.py --variants base-nothink
"""
import argparse
import os
import sys
import time

sys.path.insert(0, ".")
if os.path.isdir("/mnt/volume_d2wey28/hf_cache"):
    os.environ.setdefault("HF_HOME", "/mnt/volume_d2wey28/hf_cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/volume_d2wey28/hf_cache/hub")

from benchmark.evaluate import load_eval_configs
from benchmark.gemma_local import GemmaLocalASR, evaluate_gemma_local

BASE = "google/gemma-4-12B-it"
CODER = "yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1"

# name -> (hf repo, record-id suffix, thinking on?)
VARIANTS = {
    "base-thinking": (BASE, "thinking", True),
    "base-nothink": (BASE, "nothink", False),
    # The fine-tune is run in one flavour only, matching the base's no-thinking
    # run so the comparison isolates the fine-tune rather than the mode.
    "coder": (CODER, None, False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", help="ISO codes (default: all eval langs)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=list(VARIANTS), help="which variants to run")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true",
                    help="re-score categories that already have a result")
    args = ap.parse_args()
    langs = args.langs or list(load_eval_configs().keys())

    print("=" * 60)
    print(f"  nsanku-ASR — LLM track (local Gemma) · {len(args.variants)} variants"
          f" · {len(langs)} languages")
    print("=" * 60, flush=True)

    t_all = time.time()
    for vname in args.variants:
        model_id, label, thinking = VARIANTS[vname]
        record_id = model_id + (f"-{label}" if label else "")
        print(f"\n{'#' * 60}\n#  {vname}: {record_id}  (thinking={thinking})\n"
              f"{'#' * 60}", flush=True)
        t0 = time.time()
        try:
            runner = GemmaLocalASR(model_id, device=args.device,
                                   enable_thinking=thinking)
        except Exception as e:
            print(f"  LOAD FAILED for {model_id}: {type(e).__name__}: {e}",
                  flush=True)
            continue
        print(f"  loaded in {time.time() - t0:.0f}s", flush=True)
        try:
            for i, iso in enumerate(langs, 1):
                print(f"\n===== [{i}/{len(langs)}] {iso} =====", flush=True)
                try:
                    evaluate_gemma_local(iso, runner, model_id, label=label,
                                         force=args.force)
                except Exception as e:
                    print(f"  FAILED {iso}: {type(e).__name__}: {e}", flush=True)
        finally:
            runner.cleanup()
        print(f"\n{vname} done in {time.time() - t0:.0f}s", flush=True)

    print(f"\nAll variants done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
