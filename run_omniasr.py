"""LLM track runner — evaluate Meta's Omnilingual ASR LLM model on all eval languages.

Runs the omniASR_LLM_7B_v2 checkpoint locally on the H200 via the official
`omnilingual-asr` package (fairseq2-based). Results are written to
benchmarks_llm/{iso}.yaml alongside Gemini's entries and merged into
benchmarks/ later with merge_llm.py.

IMPORTANT — this does NOT run in the main .venv: fairseq2 pins torch 2.8 while
the main venv runs torch 2.11. One-time env setup on the H200:

    python3 -m venv /mnt/volume_d2wey28/projects/nsanku-ASR/.venv-omniasr
    /mnt/volume_d2wey28/projects/nsanku-ASR/.venv-omniasr/bin/pip install \
        torch==2.8.0 torchaudio==2.8.0
    /mnt/volume_d2wey28/projects/nsanku-ASR/.venv-omniasr/bin/pip install \
        "fairseq2==0.6" "omnilingual-asr==0.2.0" datasets soundfile pyyaml huggingface_hub

Run:  .venv-omniasr/bin/python run_omniasr.py 2>&1 | tee /tmp/nsanku_omniasr.log
      .venv-omniasr/bin/python run_omniasr.py --langs dag ewe --force
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir("/mnt/volume_d2wey28/hf_cache"):
    os.environ.setdefault("HF_HOME", "/mnt/volume_d2wey28/hf_cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/volume_d2wey28/hf_cache/hub")

from benchmark.evaluate import load_eval_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", help="ISO codes (default: all eval langs)")
    ap.add_argument("--force", action="store_true",
                    help="re-run languages even if already scored")
    args = ap.parse_args()
    langs = args.langs or list(load_eval_configs().keys())

    from benchmark.omniasr import evaluate_omniasr, drop_shared_model

    print("=" * 60)
    print(f"  nsanku-ASR — LLM track (OmniASR LLM) · {len(langs)} languages")
    print("=" * 60)
    t0 = time.time()
    done = failed = 0
    for i, iso in enumerate(langs, 1):
        print(f"\n===== [{i}/{len(langs)}] {iso} =====", flush=True)
        try:
            evaluate_omniasr(iso, force=args.force)
            done += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}", flush=True)
    drop_shared_model()
    print(f"\nLLM track done in {time.time() - t0:.0f}s "
          f"({done} ok, {failed} failed)", flush=True)


if __name__ == "__main__":
    main()
