"""Evaluation recipe for katrintomanek/whisper-small_twi-standardspeech_v2.

A Whisper-small fine-tuned on Twi standard speech. The repo ships only
`config.json` + `model.safetensors` — it has NO `preprocessor_config.json` or
tokenizer files — so the generic wrapper's `AutoProcessor` cannot load it. The
processor therefore comes from the vanilla `openai/whisper-small`, and the
fine-tuned weights from this repo.

The Akan/Twi token is forced through `forced_decoder_ids` so decoding always
emits Twi rather than relying on auto-detection.

Edit this file and open a PR at https://github.com/GhanaNLP/nsanku-ASR to change
how this model is evaluated. `build_wrapper(device)` is what the benchmark calls.
"""

import numpy as np
import torch
import transformers

from benchmark.models import _hf_auth_kwargs, cleanup_gpu

MODEL = "katrintomanek/whisper-small_twi-standardspeech_v2"
# The repo ships no feature-extractor/tokenizer; use the base small ones.
BASE = "openai/whisper-small"
SAMPLE_RATE = 16000


class TwiWhisperSmall:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.processor = transformers.WhisperProcessor.from_pretrained(
            BASE, **_hf_auth_kwargs())
        self.model = transformers.WhisperForConditionalGeneration.from_pretrained(
            MODEL, low_cpu_mem_usage=True, **_hf_auth_kwargs()
        ).to(device).eval()
        tw_tok = self.processor.tokenizer.convert_tokens_to_ids("<|tw|>")
        transcribe_tok = self.processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_tok = self.processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.forced_decoder_ids = [(1, tw_tok), (2, transcribe_tok), (3, notimestamps_tok)]
        self.model.config.forced_decoder_ids = None
        self.model.generation_config.forced_decoder_ids = None

    @torch.no_grad()
    def transcribe_batch(self, audio_arrays, sample_rate=SAMPLE_RATE, progress_cb=None):
        results = []
        for i, arr in enumerate(audio_arrays):
            if isinstance(arr, np.ndarray):
                arr = arr.astype(np.float32)
            if arr.ndim > 1:
                arr = arr.squeeze()
            input_features = self.processor(
                arr, sampling_rate=sample_rate, return_tensors="pt"
            ).input_features.to(self.device)
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=self.forced_decoder_ids,
                num_beams=1,
                do_sample=False,
            )
            text = self.processor.decode(
                predicted_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            results.append(text)
            if progress_cb and (i + 1) % 50 == 0:
                progress_cb(i + 1, len(audio_arrays))
        return results

    def cleanup(self):
        del self.model
        del self.processor
        cleanup_gpu()


def build_wrapper(device="cuda:0"):
    return TwiWhisperSmall(device=device)
