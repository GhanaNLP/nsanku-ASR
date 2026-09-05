"""Evaluation recipe for ghananlpcommunity/whisper-large-v3-turbo-twi-ctranslate.

An FP16 CTranslate2 conversion of katrintomanek/whisper-large-v3-turbo_Akan_
standardspeech_specaugment, published under the GhanaNLP org. The weights are
stored as a CTranslate2 `model.bin`, so they cannot be loaded by transformers —
they must be run through `faster-whisper` (the CTranslate2 runtime), which the
model card documents. We drive `faster_whisper.WhisperModel` directly, forcing
the language so decoding never falls back to auto-detection.

Whisper has no Twi/Akan language code — faster-whisper rejects `language="tw"`
outright ("'tw' is not a valid language code"), which is what made this model
error out on every category. The upstream fine-tune
(katrintomanek/whisper-large-v3-turbo_Akan_standardspeech_specaugment, of which
this is the CT2 conversion) reuses the Yoruba slot for Twi — its
`generation_config.json` records `language: "yo"` — so `yo` is the code to pass.

Edit this file and open a PR at https://github.com/GhanaNLP/nsanku-ASR to change
how this model is evaluated. `build_wrapper(device)` is what the benchmark calls.
"""

import numpy as np

from benchmark.models import _hf_auth_kwargs, cleanup_gpu

MODEL = "ghananlpcommunity/whisper-large-v3-turbo-twi-ctranslate"
SAMPLE_RATE = 16000
# Whisper has no Twi code; this fine-tune uses the Yoruba slot (see docstring).
LANGUAGE = "yo"


class TwiWhisperCt2:
    def __init__(self, device="cuda:0"):
        from faster_whisper import WhisperModel

        # _hf_auth_kwargs not needed: the repo is public (org, no gating).
        self.model = WhisperModel(
            MODEL,
            device="cuda" if "cuda" in device else "cpu",
            compute_type="float16" if "cuda" in device else "int8",
        )

    def transcribe_batch(self, audio_arrays, sample_rate=SAMPLE_RATE, progress_cb=None):
        results = []
        for i, arr in enumerate(audio_arrays):
            if isinstance(arr, np.ndarray):
                arr = arr.astype(np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            segments, _info = self.model.transcribe(
                arr, language=LANGUAGE, task="transcribe", beam_size=1, vad_filter=False
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            results.append(text)
            if progress_cb and (i + 1) % 50 == 0:
                progress_cb(i + 1, len(audio_arrays))
        return results

    def cleanup(self):
        del self.model
        cleanup_gpu()


def build_wrapper(device="cuda:0"):
    return TwiWhisperCt2(device=device)
