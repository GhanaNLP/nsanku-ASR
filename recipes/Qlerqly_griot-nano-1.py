"""Evaluation recipe for Qlerqly/griot-nano-1.

A 153M Conformer CTC model for Ghanaian languages (Akan, Dagbani, Ewe, Ga,
Ghanaian English). It does NOT load through transformers: the repo ships its own
`src/conformer_ctc` package plus `config.json` / `vocab.json` / `model.safetensors`,
and decoding is greedy CTC over log-mel features exactly as the model card's
`inference.py` does.

Because the weights/features pipeline is bespoke, `build_wrapper` drives the
model directly: download the repo to a local dir, import `conformer_ctc` from its
`src`, and run `greedy_decode`. The benchmark calls `transcribe_batch(audio, ...)`
with 16 kHz float numpy arrays.

Edit this file and open a PR at https://github.com/GhanaNLP/nsanku-ASR to change
how this model is evaluated.
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

# The custom conformer package lives INSIDE the model repo under src/. We load
# the model code from there — it is the model author's code, not a dependency of
# this benchmark, so it must be fetched alongside the weights.
MODEL = "Qlerqly/griot-nano-1"
SAMPLE_RATE = 16000

# Where the downloaded repo (weights + src/) is cached on this machine.
CACHE_DIR = Path("/mnt/volume_d2wey28/models/griot-nano-1")


def _model_dir():
    """Local directory holding the downloaded model repo (weights + src/)."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        MODEL,
        local_dir=str(CACHE_DIR),
        allow_patterns=[
            "config.json", "vocab.json", "tokenizer.json",
            "model.safetensors", "src/**", "*.json",
        ],
    ))


def _load():
    """Return (model, id_to_token) with the repo's own conformer code on path."""
    model_dir = _model_dir()
    sys.path.insert(0, str(model_dir / "src"))
    from conformer_ctc.data import FeatureConfig, audio_to_log_mel
    from conformer_ctc.model import ConformerCTC, ConformerCTCConfig, greedy_decode
    from safetensors.torch import load_file

    config = ConformerCTCConfig(**json.loads((model_dir / "config.json").read_text()))
    vocab = {
        str(tok): int(i)
        for tok, i in json.loads((model_dir / "vocab.json").read_text()).items()
    }
    model = ConformerCTC(config)
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")))
    id_to_token = {int(i): str(tok) for tok, i in vocab.items()}
    feat = FeatureConfig(sample_rate=SAMPLE_RATE, n_mels=config.n_mels)
    return model, id_to_token, feat, vocab


class GriotCTC:
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.model, self.id_to_token, self.feat, self.vocab = _load()
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def transcribe_batch(self, audio_arrays, sample_rate=SAMPLE_RATE, progress_cb=None):
        from conformer_ctc.model import greedy_decode
        from conformer_ctc.data import audio_to_log_mel

        results = []
        pad = self.model.config.pad_id
        blank = self.model.config.blank_id
        for i, arr in enumerate(audio_arrays):
            if isinstance(arr, np.ndarray):
                arr = arr.astype(np.float32, copy=False)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            features = audio_to_log_mel(arr, int(sample_rate), self.feat)
            lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=self.device)
            output = self.model(
                torch.from_numpy(features).unsqueeze(0).to(device=self.device),
                lengths,
            )
            text = greedy_decode(
                output.logits.cpu(),
                output.output_lengths.cpu(),
                self.id_to_token,
                blank_id=blank,
                pad_id=pad,
            )[0]
            results.append(text)
            if progress_cb and (i + 1) % 50 == 0:
                progress_cb(i + 1, len(audio_arrays))
        return results

    def cleanup(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_wrapper(device="cuda:0"):
    return GriotCTC(device=device)
