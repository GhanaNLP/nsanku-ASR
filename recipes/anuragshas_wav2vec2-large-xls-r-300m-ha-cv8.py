"""Evaluation recipe for anuragshas/wav2vec2-large-xls-r-300m-ha-cv8.

Architecture: CTC (AutoModelForCTC)
Precision: fp32 (wav2vec2/xls-r/MMS conv encoders crash in bf16 on Hopper)
Benchmarked languages: hau
Status: passed - best avg WER 83.92% (avg WER+CER 67.05%)

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on the next
benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""

from benchmark.models import CTCModel

MODEL = "anuragshas/wav2vec2-large-xls-r-300m-ha-cv8"
CTC_DECODER = "greedy"


def build_wrapper(device="cuda:0", **kwargs):
    return CTCModel(MODEL, device=device, ctc_decoder=CTC_DECODER, **kwargs)
