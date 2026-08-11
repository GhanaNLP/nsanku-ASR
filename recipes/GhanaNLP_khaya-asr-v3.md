---
model: GhanaNLP/khaya-asr-v3
---

# GhanaNLP/khaya-asr-v3

| Field | Value |
|---|---|
| **Model** | `GhanaNLP/khaya-asr-v3` |
| **Owner** | GhanaNLP |
| **URL** | https://translation-api.ghananlp.org/ |
| **Architecture** | Hosted API |
| **Precision** | n/a |
| **Benchmarked languages** | ada, bwu, dag, dga, ewe, fat, gaa, gjn, gur, hau, kus, maw, nzi, twi, xon, xsm |
| **Status** | passed - best avg WER 7.05% (avg WER+CER 5.65%) |

## Inference code used

This model was run with the inference code below from [`benchmark/khaya.py`](../benchmark/khaya.py).

```python
def _encode_wav(audio_array, sample_rate=16000):
    import soundfile as sf
    buf = BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
def _transcribe(wav_bytes, khaya_code, key):
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "audio/wav"}
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(f"{API_URL}?language={khaya_code}", headers=headers,
                              data=wav_bytes, timeout=120)
            if r.status_code == 200:
                return (r.json().get("text") or "").strip()
            if r.status_code in (429, 500, 503) and attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1)); continue
            return ""
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
    return ""
```

## Update this recipe

The YAML front-matter above controls how this model is evaluated. The following overrides are supported:

- `language` — force a source language (Whisper-style seq2seq)
- `task` — `transcribe` or `translate` (seq2seq)
- `initial_prompt` — decoder prompt text (seq2seq)
- `ctc_decoder` — `greedy` for wav2vec2/MMS/Wav2Vec2-BERT

Edit the front-matter (or the inference code notes) and open a pull request at [github.com/GhanaNLP/nsanku-ASR](https://github.com/GhanaNLP/nsanku-ASR). The benchmark reads the recipe before running a model, so the next run will use your updated recipe.
