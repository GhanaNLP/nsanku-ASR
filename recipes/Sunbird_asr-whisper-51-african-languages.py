"""Evaluation recipe for Sunbird/asr-whisper-51-african-languages.

Architecture: Whisper (seq2seq), ~2B. Multilingual (51 African languages),
force-included on the board despite the single-language rule.

IMPORTANT — this model does NOT decode like a standard Whisper. Sunbird overwrote
unused Whisper language-token ids with their own African-language tokens, so
decoding must start from the right token for the language being transcribed, and
the processor must apply `do_normalize=True`. Using the generic Whisper wrapper
produces wrong-language / garbage output.

The language token is placed in the decoder prefix via `decoder_input_ids`, NOT
via `forced_decoder_ids` as the model card shows. On transformers 5.x
`WhisperGenerationMixin.generate` no longer accepts or references
`forced_decoder_ids`: it is swallowed by `**kwargs` and silently ignored, leaving
decoding to auto-detection. Nothing warns, and because this checkpoint sets no
`generation_config.language` either, the fallback is a genuine regression rather
than a harmless one — measured over Asante Twi bible clips, auto-detection scores
CER ~0.44 against ~0.26 for the forced `<|aka|>` prefix, i.e. the ignored argument
made the model look far worse than it is. `language="aka"` is not an option: the
remapped ids are not names the Whisper tokenizer knows.

Because the token depends on the language, `build_wrapper` takes `iso_code` and
the benchmark passes the eval language through to it.

Edit this file and open a PR at https://github.com/GhanaNLP/nsanku-ASR to change
how this model is evaluated. `build_wrapper(device, iso_code)` is what the
benchmark calls.
"""

import gc

import numpy as np
import torch
import transformers

from benchmark.models import _hf_auth_kwargs, cleanup_gpu

MODEL = "Sunbird/asr-whisper-51-african-languages"
SAMPLE_RATE = 16000

# Sunbird's remapped Whisper language-token ids (from the model card).
LANGUAGE_TOKENS_WHISPER = {
    "eng": 50259, "fra": 50265, "swa": 50318, "sna": 50324, "yor": 50325, "som": 50326,
    "afr": 50327, "amh": 50334, "mlg": 50349, "lin": 50353, "hau": 50354,
    "ach": 50357, "aka": 50356, "bam": 50355, "bem": 50352, "ber": 50351,
    "cgg": 50350, "dag": 50348, "dga": 50347, "ewe": 50346, "ful": 50345,
    "ibo": 50344, "kab": 50343, "kau": 50342, "kik": 50341, "kin": 50340,
    "kln": 50339, "koo": 50338, "kpo": 50337, "led": 50336, "lgg": 50335,
    "lth": 50333, "lug": 50332, "luo": 50331, "luy": 50330, "myx": 50329,
    "nbl": 50328, "nya": 50323, "nyn": 50322, "orm": 50321, "pcm": 50320,
    "ruc": 50319, "rwm": 50317, "sot": 50316, "teo": 50315, "tsn": 50314,
    "ttj": 50313, "wol": 50312, "xho": 50311, "xog": 50310, "zul": 50309,
}

# Our eval iso_639_3 -> Sunbird language code (only the eval languages Sunbird declares).
EVAL_TO_SUNBIRD = {
    "dag": "dag", "dga": "dga", "ewe": "ewe", "hau": "hau", "kpo": "kpo",
    "twi_akuapem": "aka", "twi_asante": "aka", "twi": "aka",
}


class SunbirdWhisper:
    def __init__(self, iso_code, device="cuda:0"):
        code = EVAL_TO_SUNBIRD.get(iso_code)
        if code is None:
            raise RuntimeError(f"ARCH_UNSUPPORTED: Sunbird has no token for {iso_code}")
        self.lang_tok = LANGUAGE_TOKENS_WHISPER[code]
        self.device = device
        self.processor = transformers.WhisperProcessor.from_pretrained(MODEL, **_hf_auth_kwargs())
        self.model = transformers.WhisperForConditionalGeneration.from_pretrained(
            MODEL, low_cpu_mem_usage=True, **_hf_auth_kwargs()
        ).to(device).eval()
        tok = self.processor.tokenizer
        sot_tok = tok.convert_tokens_to_ids("<|startoftranscript|>")
        transcribe_tok = tok.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_tok = tok.convert_tokens_to_ids("<|notimestamps|>")
        # The full decoder prefix, including <|startoftranscript|>: generate()
        # continues from these ids rather than predicting the language itself.
        self.decoder_input_ids = torch.tensor(
            [[sot_tok, self.lang_tok, transcribe_tok, notimestamps_tok]],
            device=device,
        )
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
                arr, sampling_rate=sample_rate, do_normalize=True, return_tensors="pt"
            ).input_features.to(self.device)
            predicted_ids = self.model.generate(
                input_features,
                decoder_input_ids=self.decoder_input_ids,
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


def build_wrapper(device="cuda:0", iso_code=None):
    return SunbirdWhisper(iso_code=iso_code, device=device)
