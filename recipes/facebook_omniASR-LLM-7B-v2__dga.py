"""Evaluation recipe for facebook/omniASR-LLM-7B-v2 — Dagaare (dga).

Track: LLM — Meta Omnilingual ASR (omniASR_LLM_7B_v2)
Scope: THIS LANGUAGE ONLY. Every eval language has its own recipe file, so
changing this one does not affect the others.

Language conditioning: dga_Latn

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how Dagaare is evaluated.
"""


LANGUAGE_NAME = 'Dagaare'

NOTES = 'Language conditioning: dga_Latn'

# def transcribe(audio_arrays, sample_rate, lang_ids):
#     """Custom transcription override (optional)."""
#     pass
