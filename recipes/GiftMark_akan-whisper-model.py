"""Evaluation recipe for GiftMark/akan-whisper-model.

Architecture: Whisper seq2seq (AutoModelForSpeechSeq2Seq)
Precision: bf16
Benchmarked languages: twi
Status: passed - best avg WER 79.41% (avg WER+CER 60.78%)

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on the next
benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""

from benchmark.models import WhisperModel

MODEL = "GiftMark/akan-whisper-model"
LANGUAGE = None
TASK = "transcribe"
INITIAL_PROMPT = None


def build_wrapper(device="cuda:0", **kwargs):
    return WhisperModel(
        MODEL, device=device, language=LANGUAGE,
        task=TASK, initial_prompt=INITIAL_PROMPT, **kwargs,
    )
