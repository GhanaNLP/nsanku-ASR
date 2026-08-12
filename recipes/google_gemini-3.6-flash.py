"""Evaluation recipe for google/gemini-3.6-flash.

Architecture: Hosted API
Precision: n/a
Benchmarked languages: ada, bwu, dag, dga, ewe, fat, gaa, gjn, gur, hau, kus, maw, nzi, twi, xon, xsm
Status: passed - best avg WER 69.68% (avg WER+CER 55.24%)

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on the next
benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""

# This is a hosted endpoint, not a checkpoint loaded through
# benchmark/models.py, and it runs on many languages. The knobs that
# matter (API language code, or the prompt for an LLM track) therefore
# live in the PER-LANGUAGE recipes next to this file:
#
#     recipes/google_gemini-3.6-flash__twi.py
#     recipes/google_gemini-3.6-flash__ewe.py
#     ...one per eval language (see generate_api_recipes.py)
#
# Edit the file for the language you care about; the others are untouched.
