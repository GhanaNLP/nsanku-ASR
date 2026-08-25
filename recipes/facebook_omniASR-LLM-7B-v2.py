"""Evaluation recipe for facebook/omniASR-LLM-7B-v2 — all languages.

Track: LLM — Meta Omnilingual ASR (omniASR_LLM_7B_v2)
Scope: ALL LANGUAGES. This is the base recipe; per-language recipes
(recipes/facebook_omniASR-LLM-7B-v2__{iso}.py) can override LANGUAGE_NAME
and NOTES for individual languages.

Language conditioning uses Meta's own lang ids (e.g. "dag_Latn") where
available; unsupported languages fall back to unconditioned decoding.

The model is a 7B wav2vec2-style encoder + transformer decoder loaded via
the `omnilingual-asr` pip package. It runs on GPU (bf16) in its own
virtualenv (.venv-omniasr) because fairseq2 pins torch 2.8 while the main
benchmark venv uses torch 2.11.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated.
"""

MODEL_ID = "facebook/omniASR-LLM-7B-v2"
MODEL_URL = "https://github.com/facebookresearch/omnilingual-asr"
MODEL_CARD = "omniASR_LLM_7B_v2"
OWNER = "facebook"
PARAMS = "7B"


def lang_id_for(iso_code):
    """Meta lang id for an eval iso, or None to decode unconditioned."""
    try:
        from benchmark.omniasr import lang_id_for as _lang_id_for
        return _lang_id_for(iso_code)
    except ImportError:
        return None
