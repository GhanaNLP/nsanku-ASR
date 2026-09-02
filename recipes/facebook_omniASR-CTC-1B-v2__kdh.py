"""Evaluation recipe for facebook/omniASR-CTC-1B-v2 — Tem (kdh).

Track: ASR (open models) — Meta Omnilingual ASR, CTC head (omniASR_CTC_1B_v2)
Scope: THIS LANGUAGE ONLY. Every eval language has its own recipe file, so
changing this one does not affect the others, and it is separate from the
recipes for the other omniASR checkpoints.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how Tem is evaluated.
"""


LANGUAGE_NAME = 'Tem'

# Language conditioning id passed to Meta's pipeline.
# The CTC head has no conditioning: the pipeline accepts `lang` for parity with
# the LLM checkpoints but decodes identically with and without it, so this stays
# None and the score reflects unconditioned decoding. Set a "{lang}_{script}"
# id here only if a future CTC checkpoint gains language conditioning.
LANG_ID = None


def lang_id_for(iso_code):
    """Return LANG_ID for this language. Override to customize."""
    return LANG_ID
