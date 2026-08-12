"""Evaluation recipe for Google/speech-recognition — Tampulma (tpm).

Track: Hosted API — Google Speech Recognition
Scope: THIS LANGUAGE ONLY. Every eval language has its own recipe file, so
changing this one does not affect the others.

`LANGUAGE_CODE` is the BCP-47 code passed to Google's speech endpoint
(`recognize_google(audio, language=...)`). Set it to None to skip this
language. Define `transcribe(pcm_bytes, sample_rate, google_code)` to
replace the API call itself.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how Tampulma is evaluated on the
next benchmark run.
"""


LANGUAGE_CODE = None  # Google has no code for Tampulma — set one to enable


# def transcribe(pcm_bytes, sample_rate, google_code):
#     from benchmark.google import _transcribe
#     return _transcribe(pcm_bytes, sample_rate, google_code)
