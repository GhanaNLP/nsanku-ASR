"""Evaluation recipe for facebook/sherpa-onnx-omniASR-CTC-300M-v2 — Konkomba (xon).

Track: ASR (open) — sherpa-onnx ONNX export of Meta omniASR CTC 300M v2
Scope: THIS LANGUAGE ONLY. Every eval language has its own recipe file, so
changing this one does not affect the others, and each model size has its own
set.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how Konkomba is evaluated.
"""


LANGUAGE_NAME = 'Konkomba'

# sherpa-onnx decoding. greedy_search is the only method the omnilingual CTC
# recognizer supports today.
DECODING_METHOD = "greedy_search"

# Threads per decoder process. Measured on 20 cores: 2 is fastest — the graph
# does not parallelise intra-op, so throughput comes from more processes, not
# more threads per process (run_sherpa.py --workers).
NUM_THREADS = 2


def postprocess(text):
    """Clean up one hypothesis before scoring. Identity by default.

    A CTC head takes no language hint, so this is the per-language knob:
    orthography normalisation for Konkomba belongs here (e.g. mapping a character
    the export emits to the one the references use). Whatever this returns is
    what gets scored and written to the transcriptions CSV.
    """
    return text
