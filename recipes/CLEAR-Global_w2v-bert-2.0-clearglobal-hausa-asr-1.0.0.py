"""Evaluation recipe for CLEAR-Global/w2v-bert-2.0-clearglobal-hausa-asr-1.0.0.

Architecture: CTC (AutoModelForCTC)
Precision: fp32 (wav2vec2/xls-r/MMS conv encoders crash in bf16 on Hopper)
Benchmarked languages: hau
Status: passed - best avg WER 67.16% (avg WER+CER 54.82%)

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on the next
benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""

from benchmark.models import CTCModel

MODEL = "CLEAR-Global/w2v-bert-2.0-clearglobal-hausa-asr-1.0.0"
CTC_DECODER = "greedy"


def build_wrapper(device="cuda:0", **kwargs):
    return CTCModel(MODEL, device=device, ctc_decoder=CTC_DECODER, **kwargs)
