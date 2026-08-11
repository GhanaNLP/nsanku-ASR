"""Evaluation recipe for PrinceAlhassanNasamu/kusaal-whisper-small-lora.

Architecture: Whisper seq2seq (AutoModelForSpeechSeq2Seq)
Precision: bf16
Benchmarked languages: kus
Status: failed to produce valid output

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on the next
benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""

from benchmark.models import WhisperModel

MODEL = "PrinceAlhassanNasamu/kusaal-whisper-small-lora"
LANGUAGE = None
TASK = "transcribe"
INITIAL_PROMPT = None


def build_wrapper(device="cuda:0", **kwargs):
    return WhisperModel(
        MODEL, device=device, language=LANGUAGE,
        task=TASK, initial_prompt=INITIAL_PROMPT, **kwargs,
    )
