"""Evaluation recipe for facebook/omniASR-LLM-7B-v2 — Kusaal (kus).

Track: LLM — Meta Omnilingual ASR (omniASR_LLM_7B_v2)
Scope: THIS LANGUAGE ONLY. Every eval language has its own recipe file, so
changing this one does not affect the others.

Language conditioning: kus_Latn

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how Kusaal is evaluated.
"""


LANGUAGE_NAME = 'Kusaal'

NOTES = 'Language conditioning: kus_Latn'

# def transcribe(audio_arrays, sample_rate, lang_ids):
#     """Custom transcription override (optional)."""
#     pass
