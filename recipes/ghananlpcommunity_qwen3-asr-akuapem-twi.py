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

# Qwen3-ASR can be told which language to transcribe, but only accepts
# its own ~30 supported language NAMES — none of them Ghanaian — so this
# stays None (auto-detect) and the fine-tune's own bias carries it.
LANGUAGE = None
MAX_NEW_TOKENS = 256


def build_wrapper(device="cuda:0", **kwargs):
    return Qwen3ASRModel(
        MODEL, device=device, language=LANGUAGE,
        max_new_tokens=MAX_NEW_TOKENS, **kwargs,
    )
