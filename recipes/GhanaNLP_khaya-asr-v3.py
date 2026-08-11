"""Evaluation recipe for GhanaNLP/khaya-asr-v3.

Architecture: Hosted API
Precision: n/a
Benchmarked languages: ada, bwu, dag, dga, ewe, fat, gaa, gjn, gur, hau, kus, maw, nzi, twi, xon, xsm
Status: passed - best avg WER 7.05% (avg WER+CER 5.65%)

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on the next
benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""

# This model is a hosted API (benchmark/khaya.py), not run through
# benchmark/models.py. Add a build_wrapper() here only if you want to
# replace the hosted-API evaluation.

if False:
    from benchmark.khaya import _encode_wav, _transcribe
