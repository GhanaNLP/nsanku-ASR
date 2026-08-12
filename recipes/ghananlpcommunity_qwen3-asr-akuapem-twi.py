"""Evaluation recipe for ghananlpcommunity/qwen3-asr-akuapem-twi.

Architecture: Qwen3-ASR (Qwen3ASRForConditionalGeneration)
Precision: bf16
Benchmarked languages: twi
Status: not yet benchmarked (queued for the next run)

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on the next
benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""

from benchmark.models import Qwen3ASRModel

MODEL = "ghananlpcommunity/qwen3-asr-akuapem-twi"

# Requires the `qwen-asr` package (pip install qwen-asr). This checkpoint is
# saved in that library's layout, not HuggingFace's: loading it through
# transformers' Qwen3ASRForConditionalGeneration matches only 393 of 708 tensors
# and silently random-initialises the rest, so it would score as noise.

# Qwen3-ASR can be told which language to transcribe, but only accepts its own
# ~30 supported language NAMES — none of them Ghanaian — so this stays None
# (auto-detect) and the fine-tune's own bias carries it.
LANGUAGE = None

# Optional free-text hint passed to every utterance; "" means no hint.
CONTEXT = ""

MAX_NEW_TOKENS = 256
BATCH_SIZE = 8


def build_wrapper(device="cuda:0", **kwargs):
    return Qwen3ASRModel(
        MODEL, device=device, language=LANGUAGE, context=CONTEXT,
        max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE, **kwargs,
    )
