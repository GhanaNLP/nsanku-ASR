"""Evaluation recipe for Google/speech-recognition.

Architecture: Hosted API
Precision: n/a
Benchmarked languages: twi
Status: passed - best avg WER 49.77% (avg WER+CER 38.76%)

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on the next
benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""

# This is a hosted endpoint, not a checkpoint loaded through
# benchmark/models.py, and it runs on many languages. The knobs that
# matter (API language code, or the prompt for an LLM track) therefore
# live in the PER-LANGUAGE recipes next to this file:
#
#     recipes/Google_speech-recognition__twi.py
#     recipes/Google_speech-recognition__ewe.py
#     ...one per eval language (see generate_api_recipes.py)
#
# Edit the file for the language you care about; the others are untouched.
