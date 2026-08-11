"""Evaluation recipe for KhayaAI/w2v-bert-maw.

Architecture: CTC (AutoModelForCTC)
Precision: fp32 (wav2vec2/xls-r/MMS conv encoders crash in bf16 on Hopper)
Benchmarked languages: maw
Status: passed - best avg WER 16.65% (avg WER+CER 11.24%)

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on the next
benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""

from benchmark.models import CTCModel

MODEL = "KhayaAI/w2v-bert-maw"
CTC_DECODER = "greedy"


def build_wrapper(device="cuda:0", **kwargs):
    return CTCModel(MODEL, device=device, ctc_decoder=CTC_DECODER, **kwargs)
