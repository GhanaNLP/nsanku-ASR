"""Evaluation recipe for facebook/omniASR-LLM-7B-v2 — Kasem (xsm).

Track: LLM — Meta Omnilingual ASR (omniASR_LLM_7B_v2)
Scope: THIS LANGUAGE ONLY. Every eval language has its own recipe file, so
changing this one does not affect the others.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how Kasem is evaluated.
"""


LANGUAGE_NAME = 'Kasem'

# Language conditioning id passed to Meta's pipeline.
# Set to None to decode unconditioned (no language hint).
# Supported ids: https://github.com/facebookresearch/omnilingual-asr
# e.g. "dag_Latn", "ewe_Latn", "aka_Latn" (Twi), "hau_Latn", ...
LANG_ID = "xsm_Latn"


def lang_id_for(iso_code):
    """Return LANG_ID for this language. Override to customize."""
    return LANG_ID
