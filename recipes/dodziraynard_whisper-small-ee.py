"""Evaluation recipe for dodziraynard/whisper-small-ee.

Architecture: Whisper seq2seq (AutoModelForSpeechSeq2Seq)
Precision: bf16
Benchmarked languages: ewe
Status: passed - best avg WER 77.12% (avg WER+CER 52.57%)

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on the next
benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""

from benchmark.models import WhisperModel

MODEL = "dodziraynard/whisper-small-ee"
LANGUAGE = None
TASK = "transcribe"
INITIAL_PROMPT = None


def build_wrapper(device="cuda:0", **kwargs):
    return WhisperModel(
        MODEL, device=device, language=LANGUAGE,
        task=TASK, initial_prompt=INITIAL_PROMPT, **kwargs,
    )
