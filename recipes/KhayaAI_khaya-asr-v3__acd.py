"""Evaluation recipe for KhayaAI/khaya-asr-v3 — Gikyode (acd).

Track: Hosted API — Khaya (GhanaNLP)
Scope: THIS LANGUAGE ONLY. Every eval language has its own recipe file, so
changing this one does not affect the others.

`LANGUAGE_CODE` is the Khaya API's language parameter
(POST /asr/v3/transcribe?language=<code>). Set it to None to skip this
language. Define `transcribe(wav_bytes, khaya_code, key)` to replace the
API call itself.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how Gikyode is evaluated on the
next benchmark run.
"""


LANGUAGE_CODE = None  # Khaya has no code for Gikyode yet — set one to enable


# def transcribe(wav_bytes, khaya_code, key):
#     from benchmark.khaya import _transcribe
#     return _transcribe(wav_bytes, khaya_code, key)
