"""Open ASR track runner — sherpa-onnx (ONNX/CPU) exports of omniASR CTC.

Scores the ONNX conversions of Meta's omniASR CTC checkpoints. They run on CPU
with no GPU and no fairseq2, which is the point: the score says what the model
does on hardware anyone has. Results land in benchmarks/{iso}.yaml in the open
ASR track alongside the other downloadable models.

IMPORTANT — this does NOT run in the main .venv. One-time setup:

    python3 -m venv /mnt/volume_d2wey28/projects/nsanku-ASR/.venv-sherpa
    .venv-sherpa/bin/pip install sherpa-onnx "datasets<4" soundfile pyyaml \
        jiwer huggingface_hub numpy

And the model files, extracted under $SHERPA_MODEL_DIR (default
/mnt/volume_d2wey28/models/sherpa):

    huggingface-cli download \
        michsethowusu/sherpa-onnx-omnilingual-asr-1600-languages-ctc-v2 \
        <variant>.tar.bz2 --local-dir $SHERPA_MODEL_DIR
    tar xjf <variant>.tar.bz2

Run:  .venv-sherpa/bin/python run_sherpa.py --variant 300m-v2-fp32
      .venv-sherpa/bin/python run_sherpa.py --variant 300m-v2-int8 --langs dag ewe
      .venv-sherpa/bin/python run_sherpa.py --all-variants
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
from benchmark.sherpa import MODELS, DEFAULT_WORKERS, DEFAULT_NUM_THREADS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(MODELS), help="which export to score")
    ap.add_argument("--all-variants", action="store_true",
                    help="score every variant in turn")
    ap.add_argument("--langs", nargs="+", help="ISO codes (default: all eval langs)")
    ap.add_argument("--force", action="store_true",
                    help="re-run languages even if already scored")
    # Throughput comes from many small workers, not one wide one — the graph
    # does not parallelise intra-op (see benchmark/sherpa.py).
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"decoder processes (default {DEFAULT_WORKERS})")
    ap.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS,
                    help=f"threads per process (default {DEFAULT_NUM_THREADS})")
    args = ap.parse_args()

    if not args.variant and not args.all_variants:
        ap.error("pass --variant or --all-variants")
    variants = sorted(MODELS) if args.all_variants else [args.variant]
    langs = args.langs or list(load_eval_configs().keys())

    from benchmark.sherpa import evaluate_sherpa

    print("=" * 60)
    print(f"  nsanku-ASR — sherpa-onnx (CPU) · {len(variants)} variant(s) "
          f"· {len(langs)} languages")
    print("=" * 60, flush=True)
    t0 = time.time()
    for variant in variants:
        done = failed = 0
        for i, iso in enumerate(langs, 1):
            print(f"\n===== [{variant}] [{i}/{len(langs)}] {iso} =====", flush=True)
            try:
                evaluate_sherpa(iso, variant, workers=args.workers,
                                num_threads=args.num_threads, force=args.force)
                done += 1
            except Exception as e:
                failed += 1
                print(f"  FAILED: {e}", flush=True)
        print(f"\n{variant}: {done} ok, {failed} failed", flush=True)
    print(f"\nsherpa-onnx track done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
