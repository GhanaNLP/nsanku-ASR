"""ASR model wrappers for benchmarking.

Handles Whisper (seq2seq) and wav2vec2/MMS (CTC) models.
Each wrapper: load → transcribe batch → cleanup GPU memory.
"""

import os
import gc
import torch
import numpy as np
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModelForCTC,
    Wav2Vec2Processor,
)
from .config import TORCH_DTYPE, HF_TOKEN


def _hf_auth_kwargs():
    """Return kwargs for from_pretrained with HF token if available."""
    kw = {"token": HF_TOKEN} if HF_TOKEN else {}
    return kw


def resolve_batch_size(params_str=""):
    from .config import BATCH_SIZE
    s = (params_str or "").lower()
    if "b" in s:
        n = float(s.replace("b", ""))
        return BATCH_SIZE["xlarge"] if n > 3 else BATCH_SIZE["large"]
    if "m" in s:
        n = float(s.replace("m", ""))
        return BATCH_SIZE["medium"] if n > 500 else BATCH_SIZE["small"]
    return BATCH_SIZE["tiny"]


def is_ctc_model(model_id):
    id_lower = model_id.lower()
    return any(kw in id_lower for kw in [
        "wav2vec", "w2v-bert", "w2v_bert", "wav2vec2-bert", "w2v2",
        "mms-", "hubert", "xls-r", "xlsr", "data2vec",
    ])


def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()


class WhisperModel:
    """Whisper-based ASR (seq2seq). Processes samples sequentially with no_grad."""

    def __init__(self, model_id, device="cuda:0"):
        self.model_id = model_id
        self.device = device
        torch_dtype = getattr(torch, TORCH_DTYPE) if isinstance(TORCH_DTYPE, str) else TORCH_DTYPE
        self.dtype = torch_dtype

        attn = "eager"

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **_hf_auth_kwargs())
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn,
            trust_remote_code=True,
            **_hf_auth_kwargs(),
        ).to(device)

        self.model.config.forced_decoder_ids = None
        self.model.generation_config.forced_decoder_ids = None
        self.gen_kwargs = {"task": "transcribe", "return_timestamps": False}

        self.batch_size = resolve_batch_size(getattr(self.model.config, "params", ""))

    @torch.no_grad()
    def transcribe_batch(self, audio_arrays, sample_rate=16000, progress_cb=None):
        results = []
        for i, arr in enumerate(audio_arrays):
            if isinstance(arr, np.ndarray):
                arr = arr.astype(np.float32)
            if arr.ndim > 1:
                arr = arr.squeeze()

            inputs = self.processor(
                arr, sampling_rate=sample_rate, return_tensors="pt"
            ).to(self.device)

            predicted_ids = self.model.generate(
                inputs.input_features.to(self.dtype),
                **self.gen_kwargs,
            )
            text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            results.append(text)

            if progress_cb and (i + 1) % 50 == 0:
                progress_cb(i + 1, len(audio_arrays))
        return results

    def cleanup(self):
        del self.model
        del self.processor
        cleanup_gpu()


class CTCModel:
    """CTC-based ASR (wav2vec2/MMS/HuBERT)."""

    def __init__(self, model_id, device="cuda:0"):
        self.model_id = model_id
        self.device = device
        # wav2vec2/xls-r/MMS raw-waveform conv encoders trigger a broken cuDNN
        # path in bf16 on Hopper (CUDNN_STATUS_NOT_INITIALIZED). fp32 + cuDNN
        # disabled is stable and these models are small enough.
        self.dtype = torch.float32
        torch_dtype = self.dtype

        # AutoProcessor resolves the right processor (Wav2Vec2 / Wav2Vec2Bert / SeamlessM4T
        # feature extractor); fall back to Wav2Vec2Processor for older CTC repos.
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **_hf_auth_kwargs())
        except Exception:
            self.processor = Wav2Vec2Processor.from_pretrained(model_id, trust_remote_code=True, **_hf_auth_kwargs())
        self.model = AutoModelForCTC.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **_hf_auth_kwargs(),
        ).to(device)

        self.batch_size = resolve_batch_size()

    @torch.no_grad()
    def transcribe_batch(self, audio_arrays, sample_rate=16000, progress_cb=None):
        results = []
        for i, arr in enumerate(audio_arrays):
            if isinstance(arr, np.ndarray):
                arr = arr.astype(np.float32)
            if arr.ndim > 1:
                arr = arr.squeeze()

            inputs = self.processor(
                arr, sampling_rate=sample_rate, return_tensors="pt"
            ).to(self.device)

            # Wav2Vec2 uses input_values; Wav2Vec2-BERT/SeamlessM4T use input_features.
            feats = getattr(inputs, "input_values", None)
            if feats is None:
                feats = inputs.input_features
            logits = self.model(feats.to(self.dtype)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            text = self.processor.batch_decode(predicted_ids)[0].strip()
            results.append(text)

            if progress_cb and (i + 1) % 50 == 0:
                progress_cb(i + 1, len(audio_arrays))
        return results

    def cleanup(self):
        del self.model
        del self.processor
        cleanup_gpu()


def _hf_login():
    """Authenticate with HuggingFace if HF_TOKEN is set."""
    if HF_TOKEN:
        try:
            from huggingface_hub import login
            login(token=HF_TOKEN, add_to_git_credential=False)
        except Exception:
            pass


def load_asr_model(model_id, device="cuda:0"):
    """Load model, auto-detecting architecture. Returns wrapper or None on failure.

    Raises with descriptive error so caller can categorize pass/fail.
    """
    torch.backends.cudnn.enabled = False
    _hf_login()
    try:
        if is_ctc_model(model_id):
            return CTCModel(model_id, device=device)
        return WhisperModel(model_id, device=device)
    except Exception as e:
        err = str(e)
        if "gated repo" in err.lower() or "403" in err:
            raise RuntimeError(f"GATED: {err}")
        if "does not support" in err.lower() and "attention" in err.lower():
            raise RuntimeError(f"ARCH_UNSUPPORTED: {err}")
        if "unrecognized configuration class" in err.lower() or "unrecognized processing class" in err.lower():
            raise RuntimeError(f"ARCH_UNKNOWN: {err}")
        raise
