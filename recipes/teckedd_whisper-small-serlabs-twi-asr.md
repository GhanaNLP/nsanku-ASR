# teckedd/whisper-small-serlabs-twi-asr

| Field | Value |
|---|---|
| **Model** | `teckedd/whisper-small-serlabs-twi-asr` |
| **Owner** | teckedd |
| **URL** | https://huggingface.co/teckedd/whisper-small-serlabs-twi-asr |
| **Architecture** | Whisper seq2seq (AutoModelForSpeechSeq2Seq) |
| **Precision** | bf16 |
| **Benchmarked languages** | twi |
| **Status** | passed - best avg WER 101.74% |

## Inference code used

This model was run with the inference code below from [`benchmark/models.py`](../benchmark/models.py).

```python
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

```

## Update this recipe

If this model needs custom inference (custom tokenizer / processor, language decoding, LM post-processing, etc.) and you believe the WER above is not representative, edit this file and open a pull request at [github.com/GhanaNLP/nsanku-ASR](https://github.com/GhanaNLP/nsanku-ASR). The benchmark will use your updated recipe on the next run.
