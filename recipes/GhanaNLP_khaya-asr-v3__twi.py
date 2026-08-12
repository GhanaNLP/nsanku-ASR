"""Evaluation recipe for GhanaNLP/khaya-asr-v3 — Twi (twi).

Track: Hosted API — Khaya (GhanaNLP)
Scope: THIS LANGUAGE ONLY. Every eval language has its own recipe file, so
changing this one does not affect the others.

`LANGUAGE_CODE` is the Khaya API's language parameter
(POST /asr/v3/transcribe?language=<code>). Set it to None to skip this
language. Define `transcribe(wav_bytes, khaya_code, key)` to replace the
API call itself.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how Twi is evaluated on the
next benchmark run.
"""


# 'atw' (Akuapem Twi) was the original choice and wins on the bible category
# (WER 8.93% vs 34.84%), but the twi eval set is mixed-dialect and 'atw' loses
# badly on jw (88.69% vs 49.52%). 'twi' is the general-Twi endpoint.
LANGUAGE_CODE = 'twi'


# def transcribe(wav_bytes, khaya_code, key):
#     from benchmark.khaya import _transcribe
#     return _transcribe(wav_bytes, khaya_code, key)
