"""Evaluation recipe for cdli/whisper-small_finetuned_ghanian_ga_standard_speech_v1.0.

Architecture: Whisper seq2seq (AutoModelForSpeechSeq2Seq)
Precision: bf16
Benchmarked languages: gaa
Status: passed - best avg WER 111.58% (avg WER+CER 82.73%)

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on the next
benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""

from benchmark.models import WhisperModel

MODEL = "cdli/whisper-small_finetuned_ghanian_ga_standard_speech_v1.0"
LANGUAGE = None
TASK = "transcribe"
INITIAL_PROMPT = None


def build_wrapper(device="cuda:0", **kwargs):
    return WhisperModel(
        MODEL, device=device, language=LANGUAGE,
        task=TASK, initial_prompt=INITIAL_PROMPT, **kwargs,
    )
