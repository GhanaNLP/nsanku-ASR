"""Evaluation recipe for google/gemini-3.6-flash — Birifor_Southern (biv).

Track: LLM — Gemini (gemini-3.6-flash)
Scope: THIS LANGUAGE ONLY. Every eval language has its own recipe file, so
changing this one does not affect the others.

`PROMPT` is the exact prompt sent with each audio clip. This is the main
knob for an LLM track — tune the wording, orthography hints, or examples
for this language. The transcription is read back out of the square
brackets (see `benchmark.gemini._parse`), so keep that instruction unless
you also override `transcribe(wav_bytes, prompt)`.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how Birifor_Southern is evaluated on the
next benchmark run.
"""


LANGUAGE_NAME = 'Birifor_Southern'

PROMPT = (
    "Transcribe the speech in this audio exactly as spoken. "
    "The language is Birifor_Southern. "
    "Put the transcription inside square brackets, e.g. [the man went to the market]. "
    "Output ONLY the bracketed transcription, nothing else."
)


# def transcribe(wav_bytes, prompt):
#     from benchmark.gemini import _transcribe
#     return _transcribe(wav_bytes, prompt)
