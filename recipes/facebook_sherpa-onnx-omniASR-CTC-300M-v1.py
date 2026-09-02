"""Evaluation recipe for facebook/sherpa-onnx-omniASR-CTC-300M-v1 — all languages.

Track: ASR (open models) — sherpa-onnx ONNX export of Meta omniASR CTC 300M v2
Scope: ALL LANGUAGES. This is the base recipe; the per-language recipes
(recipes/facebook_sherpa-onnx-omniASR-CTC-300M-v1__{iso}.py) are what a run actually reads.

This is the unquantised ONNX conversion of the omniASR CTC checkpoint, decoded
with sherpa-onnx on CPU — no GPU and no fairseq2. That is why it is benchmarked
separately from the fairseq2 run in benchmark/omniasr.py: the score here
describes the model on hardware anyone has. The name carries no precision
claim; the export does not state one and sherpa-onnx picks the runtime dtype.

Source weights: https://huggingface.co/csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12
Variant key:    300m-v1

    .venv-sherpa/bin/python run_sherpa.py --variant 300m-v1

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated.
"""

MODEL_ID = "facebook/sherpa-onnx-omniASR-CTC-300M-v1"
MODEL_URL = "https://huggingface.co/csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12"
OWNER = "facebook"
PARAMS = "300M"
MODEL_CLASS = "non-llm"
ONNX_FILE = "model.onnx"
DECODING_METHOD = "greedy_search"
