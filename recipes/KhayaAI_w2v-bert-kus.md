---
model: KhayaAI/w2v-bert-kus
ctc_decoder: greedy
---

# KhayaAI/w2v-bert-kus

| Field | Value |
|---|---|
| **Model** | `KhayaAI/w2v-bert-kus` |
| **Owner** | KhayaAI |
| **URL** | https://huggingface.co/KhayaAI/w2v-bert-kus |
| **Architecture** | CTC (AutoModelForCTC) |
| **Precision** | fp32 (wav2vec2/xls-r/MMS conv encoders crash in bf16 on Hopper) |
| **Benchmarked languages** | kus |
| **Status** | passed - best avg WER 37.90% (avg WER+CER 27.61%) |

## Inference code used

This model was run with the inference code below from [`benchmark/models.py`](../benchmark/models.py).

```python
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

```

## Update this recipe

The YAML front-matter above controls how this model is evaluated. The following overrides are supported:

- `language` — force a source language (Whisper-style seq2seq)
- `task` — `transcribe` or `translate` (seq2seq)
- `initial_prompt` — decoder prompt text (seq2seq)
- `ctc_decoder` — `greedy` for wav2vec2/MMS/Wav2Vec2-BERT

Edit the front-matter (or the inference code notes) and open a pull request at [github.com/GhanaNLP/nsanku-ASR](https://github.com/GhanaNLP/nsanku-ASR). The benchmark reads the recipe before running a model, so the next run will use your updated recipe.
