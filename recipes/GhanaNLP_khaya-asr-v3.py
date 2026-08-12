"""Evaluation recipe for GhanaNLP/khaya-asr-v3.

Architecture: Hosted API
Precision: n/a
Benchmarked languages: ada, bwu, dag, dga, ewe, fat, gaa, gjn, gur, hau, kus, maw, nzi, twi, xon, xsm
Status: passed - best avg WER 7.05% (avg WER+CER 5.65%)

Khaya is one API evaluated on many languages, so the knobs that matter live in
the PER-LANGUAGE recipes instead of this file:

    recipes/GhanaNLP_khaya-asr-v3__twi.py   ->  LANGUAGE_CODE = 'atw'
    recipes/GhanaNLP_khaya-asr-v3__ewe.py   ->  LANGUAGE_CODE = 'ewe'
    ...one per eval language

Edit the file for the language you care about and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR — the other languages are untouched.
"""

# The hosted-API evaluation lives in benchmark/khaya.py; per-language behaviour
# is configured in the __{iso}.py recipes next to this file.
