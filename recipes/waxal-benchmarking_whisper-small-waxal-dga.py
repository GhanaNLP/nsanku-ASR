"""Evaluation recipe for waxal-benchmarking/whisper-small-waxal-dga.

Architecture: Whisper seq2seq (AutoModelForSpeechSeq2Seq)
Precision: bf16
Benchmarked languages: dga
Status: passed - best avg WER 59.53% (avg WER+CER 44.62%)

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on the next
benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""

from benchmark.models import WhisperModel

MODEL = "waxal-benchmarking/whisper-small-waxal-dga"
LANGUAGE = None
TASK = "transcribe"
INITIAL_PROMPT = None


def build_wrapper(device="cuda:0", **kwargs):
    return WhisperModel(
        MODEL, device=device, language=LANGUAGE,
        task=TASK, initial_prompt=INITIAL_PROMPT, **kwargs,
    )
