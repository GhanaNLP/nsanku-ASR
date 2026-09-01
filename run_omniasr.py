"""Runner for Meta's Omnilingual ASR checkpoints across all eval languages.

Runs a checkpoint locally on the H200 via the official `omnilingual-asr`
package (fairseq2-based). Results are written to benchmarks_llm/{iso}.yaml
alongside Gemini's entries and merged into benchmarks/ later with merge_llm.py.

Two checkpoint families share this runner and the same inference pipeline, but
land in different leaderboard tracks:
  * omniASR_LLM_*  — LLM decoder, track "llm" (the default)
  * omniASR_CTC_*  — plain CTC head, track "non-llm" (open ASR)
Pick with --preset ctc-1b, or set --card/--model-id/--params/--model-class.

IMPORTANT — this does NOT run in the main .venv: fairseq2 pins torch 2.8 while
the main venv runs torch 2.11. One-time env setup on the H200:

    python3 -m venv /mnt/volume_d2wey28/projects/nsanku-ASR/.venv-omniasr
    /mnt/volume_d2wey28/projects/nsanku-ASR/.venv-omniasr/bin/pip install \
        torch==2.8.0 torchaudio==2.8.0
    /mnt/volume_d2wey28/projects/nsanku-ASR/.venv-omniasr/bin/pip install \
        "fairseq2==0.6" "omnilingual-asr==0.2.0" datasets soundfile pyyaml huggingface_hub

Run:  .venv-omniasr/bin/python run_omniasr.py 2>&1 | tee /tmp/nsanku_omniasr.log
      .venv-omniasr/bin/python run_omniasr.py --langs dag ewe --force
      .venv-omniasr/bin/python run_omniasr.py --preset ctc-1b
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

# Checkpoint presets: fairseq2 card -> how the result is published.
PRESETS = {
    "llm-7b": {"card": "omniASR_LLM_7B_v2", "model_id": "facebook/omniASR-LLM-7B-v2",
               "params": "7B", "model_class": "llm"},
    "ctc-1b": {"card": "omniASR_CTC_1B_v2", "model_id": "facebook/omniASR-CTC-1B-v2",
               "params": "1B", "model_class": "non-llm"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", help="ISO codes (default: all eval langs)")
    ap.add_argument("--force", action="store_true",
                    help="re-run languages even if already scored")
    ap.add_argument("--preset", default="llm-7b", choices=sorted(PRESETS),
                    help="checkpoint to run (default: llm-7b)")
    ap.add_argument("--card", help="override the fairseq2 model card")
    ap.add_argument("--model-id", help="override the model id shown on the board")
    ap.add_argument("--params", help="override the reported parameter count")
    ap.add_argument("--model-class", choices=["llm", "non-llm"],
                    help="override the leaderboard track")
    args = ap.parse_args()
    langs = args.langs or list(load_eval_configs().keys())

    cfg = dict(PRESETS[args.preset])
    for key in ("card", "model_id", "params", "model_class"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    from benchmark.omniasr import evaluate_omniasr, drop_shared_model

    print("=" * 60)
    print(f"  nsanku-ASR — OmniASR {cfg['card']} -> track "
          f"{cfg['model_class']} · {len(langs)} languages")
    print("=" * 60)
    t0 = time.time()
    done = failed = 0
    for i, iso in enumerate(langs, 1):
        print(f"\n===== [{i}/{len(langs)}] {iso} =====", flush=True)
        try:
            evaluate_omniasr(iso, force=args.force, **cfg)
            done += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}", flush=True)
    drop_shared_model()
    print(f"\n{cfg['card']} done in {time.time() - t0:.0f}s "
          f"({done} ok, {failed} failed)", flush=True)


if __name__ == "__main__":
    main()
