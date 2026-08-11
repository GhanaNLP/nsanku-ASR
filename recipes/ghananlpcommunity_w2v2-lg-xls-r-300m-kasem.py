"""Evaluation recipe for ghananlpcommunity/w2v2-lg-xls-r-300m-kasem.

Architecture: CTC (AutoModelForCTC)
Precision: fp32 (wav2vec2/xls-r/MMS conv encoders crash in bf16 on Hopper)
Benchmarked languages: xsm
Status: passed - best avg WER 38.88% (avg WER+CER 30.07%)

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on the next
benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""

from benchmark.models import CTCModel

MODEL = "ghananlpcommunity/w2v2-lg-xls-r-300m-kasem"
CTC_DECODER = "greedy"


def build_wrapper(device="cuda:0", **kwargs):
    return CTCModel(MODEL, device=device, ctc_decoder=CTC_DECODER, **kwargs)
