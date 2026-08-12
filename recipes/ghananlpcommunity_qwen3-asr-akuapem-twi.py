"""Evaluation recipe for ghananlpcommunity/qwen3-asr-akuapem-twi.

Architecture: Qwen3-ASR (1.7B base) — run via ONNX Runtime
Precision: fp32
Benchmarked languages: twi
Status: not yet benchmarked (queued for the next run)

This model is evaluated through ONNX, not PyTorch, for speed. The `qwen_asr`
PyTorch path produces the same transcripts but takes ~8s per sample on an H200
(single CPU core saturated, GPU idle — generation runs layer-by-layer in
Python); the exported ONNX graphs reach RTF ~0.25, about 5x faster, with
accuracy unchanged (12.48% vs 13.30% WER on the same 16 bible samples).

Export step, run once per checkpoint:

    git clone https://github.com/Wasser1462/Qwen3-ASR-onnx
    cd Qwen3-ASR-onnx
    python3 export_qwen3_asr_onnx.py \\
        --model <path to the HF snapshot> \\
        --outdir <ONNX_DIR> --device cpu --max-total-len 512 --verify

That writes conv_frontend.onnx, encoder.onnx and decoder.onnx (plus .int8
variants, which this recipe does NOT use — they are no faster, because the
bottleneck is per-token session calls rather than weight size, and on CUDA they
are measurably slower).

Set ONNX_DIR below (or the QWEN3_ASR_ONNX_DIR environment variable) to wherever
the export landed. Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on
the next benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""
import os

from benchmark.models import Qwen3ASROnnxModel

MODEL = "ghananlpcommunity/qwen3-asr-akuapem-twi"

# Where `run.sh` put conv_frontend.onnx / encoder.onnx / decoder.onnx.
ONNX_DIR = os.environ.get(
    "QWEN3_ASR_ONNX_DIR",
    "/mnt/volume_d2wey28/projects/qwen3-onnx-export/model/model_twi",
)

# Checkout of https://github.com/Wasser1462/Qwen3-ASR-onnx — its
# infer_qwen3_asr.py is the reference decoder and is called per batch.
EXPORT_REPO = os.environ.get(
    "QWEN3_ASR_EXPORT_REPO", "/mnt/volume_d2wey28/projects/qwen3-onnx-export")

BATCH_SIZE = 16
MAX_NEW_TOKENS = 100

# "encoder.int8.onnx" / "decoder.int8.onnx" are available but slower on CUDA.
ENCODER = "encoder.onnx"
DECODER = "decoder.onnx"


def build_wrapper(device="cuda:0", **kwargs):
    return Qwen3ASROnnxModel(
        MODEL, device=device, onnx_dir=ONNX_DIR, export_repo=EXPORT_REPO,
        batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS,
        encoder=ENCODER, decoder=DECODER, **kwargs,
    )
