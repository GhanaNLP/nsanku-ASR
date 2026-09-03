"""ASR model wrappers for benchmarking.

Handles Whisper (seq2seq) and wav2vec2/MMS (CTC) models.
Each wrapper: load → transcribe batch → cleanup GPU memory.
"""

import os
import gc
import re
import sys
import torch
import numpy as np
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModelForCTC,
    Wav2Vec2Processor,
)
from .config import TORCH_DTYPE, HF_TOKEN
from .recipes import load_recipe


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


# CTC model_types (config.model_type) that AutoModelForCTC handles.
CTC_MODEL_TYPES = {
    "wav2vec2", "wav2vec2-bert", "wav2vec2-conformer", "hubert", "wavlm",
    "unispeech", "unispeech-sat", "sew", "sew-d", "data2vec-audio", "mctct",
}


def _detect_arch(model_id):
    """Return 'ctc' | 'whisper' | 'qwen2audio' | 'qwen3asr' | 'seamless' | 'seq2seq'.

    Falls back to the model name only if the config can't be read.
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, **_hf_auth_kwargs())
        archs = list(getattr(cfg, "architectures", None) or [])
        mtype = (getattr(cfg, "model_type", "") or "").lower()
        if any(a.endswith("ForCTC") for a in archs) or mtype in CTC_MODEL_TYPES:
            return "ctc"
        if mtype == "whisper" or any("Whisper" in a for a in archs):
            return "whisper"
        if mtype in ("qwen2_audio", "qwen2audio") or any("Qwen2Audio" in a for a in archs):
            return "qwen2audio"
        if mtype in ("qwen3_asr", "qwen3asr") or any("Qwen3ASR" in a for a in archs):
            return "qwen3asr"
        if mtype.startswith("seamless") or any("SeamlessM4T" in a for a in archs):
            return "seamless"  # not supported as ASR here (no Ghanaian tgt_lang codes)
        return "seq2seq"
    except Exception:
        return "name:" + model_id.lower()


def is_ctc_model(model_id):
    """Legacy name-based heuristic (fallback when config is unavailable)."""
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

    def __init__(self, model_id, device="cuda:0", language=None, task="transcribe", initial_prompt=None):
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
        self.gen_kwargs = {"task": task, "return_timestamps": False}
        if language:
            self.gen_kwargs["language"] = language
        if initial_prompt:
            self.gen_kwargs["prompt"] = initial_prompt

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

    def __init__(self, model_id, device="cuda:0", ctc_decoder="greedy"):
        self.model_id = model_id
        self.device = device
        # wav2vec2/xls-r/MMS raw-waveform conv encoders trigger a broken cuDNN
        # path in bf16 on Hopper (CUDNN_STATUS_NOT_INITIALIZED). fp32 + cuDNN
        # disabled is stable and these models are small enough.
        self.dtype = torch.float32
        torch_dtype = self.dtype
        if ctc_decoder != "greedy":
            raise ValueError(f"unsupported ctc_decoder '{ctc_decoder}' (only 'greedy' is available)")
        self.decoder = "greedy"

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


class Qwen2AudioModel:
    """Audio-LLM ASR (Qwen2-Audio fine-tunes).

    Unlike CTC/Whisper models these are instruction-following multimodal LLMs:
    the audio is one turn of a chat and the transcript is generated as a reply,
    so the PROMPT is part of the model's inference contract. The defaults follow
    the prompt FarmerlineML documents on the model cards; a recipe can override
    `system_prompt` / `user_prompt` per model.
    """

    DEFAULT_SYSTEM = ("You are a speech recognition system. Transcribe the audio "
                      "exactly as spoken. Return only the transcript, nothing else.")
    DEFAULT_USER = "Transcribe this audio exactly."

    def __init__(self, model_id, device="cuda:0", system_prompt=None,
                 user_prompt=None, max_new_tokens=256):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        self.model_id = model_id
        self.device = device
        self.dtype = getattr(torch, TORCH_DTYPE)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self.user_prompt = user_prompt or self.DEFAULT_USER
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id, **_hf_auth_kwargs())
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, dtype=self.dtype, attn_implementation="sdpa",
            **_hf_auth_kwargs(),
        ).to(device)
        self.model.eval()

    def _chat_text(self):
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "audio.wav"},
                {"type": "text", "text": self.user_prompt},
            ]},
        ]
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)

    def transcribe_batch(self, audio_arrays, sample_rate=16000, progress_cb=None):
        text = self._chat_text()
        sr = getattr(self.processor.feature_extractor, "sampling_rate", sample_rate)
        results = []
        for i, arr in enumerate(audio_arrays):
            if isinstance(arr, np.ndarray):
                arr = arr.astype(np.float32)
            if arr.ndim > 1:
                arr = arr.squeeze()

            try:
                inputs = self.processor(text=text, audio=arr, sampling_rate=sr,
                                        return_tensors="pt")
            except TypeError:
                # transformers < 4.46 named the argument `audios`
                inputs = self.processor(text=text, audios=arr, sampling_rate=sr,
                                        return_tensors="pt")
            inputs = inputs.to(self.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                          do_sample=False)
            n = inputs["input_ids"].shape[1]
            results.append(
                self.processor.batch_decode(out[:, n:], skip_special_tokens=True)[0].strip()
            )

            if progress_cb and (i + 1) % 50 == 0:
                progress_cb(i + 1, len(audio_arrays))
        return results

    def cleanup(self):
        del self.model
        del self.processor
        cleanup_gpu()


class Qwen3ASROnnxModel:
    """Qwen3-ASR fine-tunes run through ONNX Runtime instead of PyTorch.

    The `qwen_asr` PyTorch path works but is unusably slow here — ~8s per sample
    on an H200, single CPU core pegged and the GPU idle, because generation runs
    layer-by-layer in Python. Exported to ONNX the same model reaches RTF ~0.25,
    roughly 5x faster, with accuracy unchanged (12.48% vs 13.30% WER on the same
    16 bible samples; the exporter's own check puts encoder/decoder max_diff at
    0.000000/0.000010).

    Export once with https://github.com/Wasser1462/Qwen3-ASR-onnx (`run.sh`,
    pointing MODEL_DIR at the HF snapshot) to produce conv_frontend.onnx,
    encoder.onnx and decoder.onnx, then point `onnx_dir` here.

    Inference shells out to that repo's `infer_qwen3_asr.py` per batch rather
    than reimplementing its decode loop — that script is the reference decoder
    and is what the numbers above were measured with. INT8 weights are NOT used
    by default: they are no faster (the bottleneck is per-token session calls,
    not weight size) and on CUDA they are slower.
    """

    LINE_RE = re.compile(r"^\[(?:.*/)?(\d+)\.wav\]\s*(?:language\s+\S+<asr_text>)?(.*)$")

    def __init__(self, model_id, device="cuda:0", onnx_dir=None, export_repo=None,
                 snapshot_dir=None, batch_size=16, max_new_tokens=100,
                 encoder="encoder.onnx", decoder="decoder.onnx"):
        import shutil
        self.model_id = model_id
        self.device = "cuda" if str(device).startswith("cuda") else "cpu"
        self.onnx_dir = onnx_dir or os.environ.get("QWEN3_ASR_ONNX_DIR", "")
        self.export_repo = export_repo or os.environ.get(
            "QWEN3_ASR_EXPORT_REPO", "/mnt/volume_d2wey28/projects/qwen3-onnx-export")
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.encoder = encoder
        self.decoder = decoder
        if not self.onnx_dir or not os.path.isdir(self.onnx_dir):
            raise RuntimeError(
                f"ONNX_MISSING: no exported model at {self.onnx_dir!r}. "
                "Run Qwen3-ASR-onnx/run.sh for this checkpoint first.")
        self.snapshot_dir = snapshot_dir or self._resolve_snapshot(model_id)
        self.script = os.path.join(self.export_repo, "infer_qwen3_asr.py")
        if not os.path.isfile(self.script):
            raise RuntimeError(f"ONNX_MISSING: exporter repo not found at {self.export_repo}")
        self._tmp = shutil  # kept only so cleanup() has something to release

    @staticmethod
    def _resolve_snapshot(model_id):
        """Local path of the HF snapshot (the decoder needs its tokenizer/config)."""
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, token=HF_TOKEN or None)

    def transcribe_batch(self, audio_arrays, sample_rate=16000, progress_cb=None):
        import subprocess, tempfile
        import soundfile as sf
        results = []
        for start in range(0, len(audio_arrays), self.batch_size):
            batch = audio_arrays[start:start + self.batch_size]
            with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or None) as tmp:
                for i, arr in enumerate(batch):
                    if isinstance(arr, np.ndarray):
                        arr = arr.astype(np.float32)
                    if arr.ndim > 1:
                        arr = arr.squeeze()
                    sf.write(os.path.join(tmp, f"{i:04d}.wav"), arr, sample_rate)
                cmd = [
                    sys.executable, self.script,
                    "--conv_frontend", os.path.join(self.onnx_dir, "conv_frontend.onnx"),
                    "--encoder", os.path.join(self.onnx_dir, self.encoder),
                    "--decoder", os.path.join(self.onnx_dir, self.decoder),
                    "--model", self.snapshot_dir,
                    "--device", self.device,
                    "--max-new-tokens", str(self.max_new_tokens),
                    "--wav",
                ] + [os.path.join(tmp, f"{i:04d}.wav") for i in range(len(batch))]
                proc = subprocess.run(cmd, cwd=self.export_repo, capture_output=True,
                                      text=True, timeout=3600)
            texts = {}
            for line in proc.stdout.splitlines():
                m = self.LINE_RE.match(line.strip())
                if m:
                    texts[int(m.group(1))] = m.group(2).strip()
            if not texts and proc.returncode != 0:
                raise RuntimeError(f"ONNX inference failed: {proc.stderr[-400:]}")
            # A single wav prints without the "[path] " prefix.
            if len(batch) == 1 and not texts:
                out = [l for l in proc.stdout.splitlines() if l and not l.startswith("RTF")]
                texts = {0: re.sub(r"^language\s+\S+<asr_text>", "", out[-1]).strip()} if out else {}
            results.extend(texts.get(i, "") for i in range(len(batch)))

            if progress_cb:
                progress_cb(min(start + self.batch_size, len(audio_arrays)),
                            len(audio_arrays))
        return results

    def cleanup(self):
        cleanup_gpu()


class Qwen3ASRModel:
    """Qwen3-ASR fine-tunes, loaded with the `qwen_asr` package (pip install qwen-asr).

    NOT loadable through transformers. These checkpoints are saved in qwen_asr's
    layout — every tensor under a `thinker.` prefix, an audio tower carrying
    proj1/proj2 — while `Qwen3ASRForConditionalGeneration` expects
    `model.language_model.*` / `model.audio_tower.*` / `lm_head.weight`. Only 393
    of 708 tensors line up, and transformers does not fail on the mismatch: it
    randomly initialises every unmatched weight and runs happily, producing
    scores from an untrained model. Hence the dedicated library, which is also
    what the model cards document.

    `language` forces a transcription language but accepts only Qwen's own ~30
    language names, none of them Ghanaian, so the default is None (auto-detect).
    `context` is an optional prompt-style hint the library supports.
    """

    def __init__(self, model_id, device="cuda:0", language=None, context="",
                 max_new_tokens=256, batch_size=8):
        from qwen_asr import Qwen3ASRModel as _QwenASRModel
        self.model_id = model_id
        self.device = device
        self.language = language
        self.context = context
        self.batch_size = batch_size
        self.model = _QwenASRModel.from_pretrained(
            model_id,
            dtype=getattr(torch, TORCH_DTYPE),
            device_map=device,
            max_new_tokens=max_new_tokens,
            max_inference_batch_size=batch_size,
        )

    def transcribe_batch(self, audio_arrays, sample_rate=16000, progress_cb=None):
        results = []
        for start in range(0, len(audio_arrays), self.batch_size):
            chunk = []
            for arr in audio_arrays[start:start + self.batch_size]:
                if isinstance(arr, np.ndarray):
                    arr = arr.astype(np.float32)
                if arr.ndim > 1:
                    arr = arr.squeeze()
                chunk.append((arr, sample_rate))

            out = self.model.transcribe(chunk, context=self.context,
                                        language=self.language)
            for t in out:
                text = getattr(t, "text", None)
                results.append((text or "").strip())

            if progress_cb:
                progress_cb(min(start + self.batch_size, len(audio_arrays)),
                            len(audio_arrays))
        return results

    def cleanup(self):
        del self.model
        cleanup_gpu()


def _hf_login():
    """Authenticate with HuggingFace if HF_TOKEN is set."""
    if HF_TOKEN:
        try:
            from huggingface_hub import login
            login(token=HF_TOKEN, add_to_git_credential=False)
        except Exception:
            pass


def load_asr_model(model_id, device="cuda:0", iso_code=None):
    """Load model, auto-detecting architecture. Returns wrapper or None on failure.

    If the model has a recipe (recipes/{owner}_{model}.py) with a
    `build_wrapper(device)`, that is used as-is — model authors can fully
    customize inference. A multilingual model's recipe may also accept
    `iso_code`, so its wrapper can pick the right per-language decoding (e.g.
    Sunbird's remapped Whisper language token). Otherwise the standard wrapper
    for the detected architecture is used. Raises with descriptive error so
    caller can categorize pass/fail.
    """
    torch.backends.cudnn.enabled = False
    _hf_login()

    recipe = load_recipe(model_id)
    if recipe is not None and hasattr(recipe, "build_wrapper"):
        import inspect
        params = inspect.signature(recipe.build_wrapper).parameters
        kwargs = {"device": device}
        # Only recipes that EXPLICITLY name iso_code receive it; recipes that
        # forward **kwargs to a wrapper constructor must not (it would crash them).
        if "iso_code" in params:
            kwargs["iso_code"] = iso_code
        wrapper = recipe.build_wrapper(**kwargs)
        if wrapper is not None:
            print(f"    Using recipe: {recipe.__name__}.build_wrapper()")
            return wrapper

    arch = _detect_arch(model_id)
    if arch == "seamless":
        # SeamlessM4T speech-to-text: these repos ship only the base seamless
        # language codes (no Ghanaian tgt_lang), so we can't transcribe reliably.
        raise RuntimeError("ARCH_UNSUPPORTED: SeamlessM4T ASR (no target-language code)")
    try:
        if arch == "qwen2audio":
            return Qwen2AudioModel(model_id, device=device)
        if arch == "qwen3asr":
            return Qwen3ASRModel(model_id, device=device)
        if arch == "ctc" or (arch.startswith("name:") and is_ctc_model(model_id)):
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
