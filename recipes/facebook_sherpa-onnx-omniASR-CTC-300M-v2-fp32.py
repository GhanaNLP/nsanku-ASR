"""Evaluation recipe for facebook/sherpa-onnx-omniASR-CTC-300M-v2-fp32 — all languages.

Track: ASR (open) — sherpa-onnx ONNX export of Meta omniASR CTC 300M v2 (fp32)
Scope: ALL LANGUAGES. This is the base recipe; the per-language recipes
(recipes/michsethowusu_sherpa-onnx-omniASR-CTC-300M-v2-fp32__{iso}.py) are what a run actually reads.

This is an ONNX conversion of the omniASR CTC checkpoint, decoded with
sherpa-onnx on CPU — no GPU and no fairseq2. That is the reason it is
benchmarked separately from the fairseq2 run in benchmark/omniasr.py: the score
here describes the model on hardware anyone has.

Source weights: https://huggingface.co/michsethowusu/sherpa-onnx-omnilingual-asr-1600-languages-ctc-v2
Variant key:    300m-v2-fp32

    .venv-sherpa/bin/python run_sherpa.py --variant 300m-v2-fp32

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated.
"""

MODEL_ID = "facebook/sherpa-onnx-omniASR-CTC-300M-v2-fp32"
MODEL_URL = "https://huggingface.co/michsethowusu/sherpa-onnx-omnilingual-asr-1600-languages-ctc-v2"
OWNER = "facebook"
PARAMS = "300M"
MODEL_CLASS = "non-llm"
ONNX_FILE = "model.onnx"
DECODING_METHOD = "greedy_search"
