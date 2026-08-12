# Model recipes

One real Python module per benchmarked model: `recipes/{owner}_{model}.py`.
The benchmark **imports and runs this exact file** before evaluating a model
(`benchmark/recipes.py` + `benchmark/models.py#load_asr_model`), so what you see
here is what runs.

## The contract

A recipe must expose `build_wrapper(device="cuda:0")`, which returns the model
wrapper used for inference. The generated recipes delegate to the standard
wrappers in `benchmark/models.py`:

- **CTC** (wav2vec2 / MMS / Wav2Vec2-BERT): `CTCModel`, with `CTC_DECODER`
- **Whisper-style seq2seq**: `WhisperModel`, with `LANGUAGE`, `TASK`,
  `INITIAL_PROMPT`

```
>>> from benchmark.models import load_asr_model
>>> model = load_asr_model("KhayaAI/w2v-bert-ada")   # uses recipes/KhayaAI_w2v-bert-ada.py
```

If a model has no recipe (or the recipe fails to import), the benchmark falls
back to the standard wrapper for the detected architecture.

## API / LLM tracks: one recipe **per language**

Khaya, Google ASR and Gemini are a single endpoint evaluated on every language,
so a single file per model would mean nobody could change one language without
touching the rest. These tracks therefore get a recipe **per language**:

```
recipes/GhanaNLP_khaya-asr-v3__twi.py      # Khaya, Twi only
recipes/Google_speech-recognition__ewe.py  # Google ASR, Ewe only
recipes/google_gemini-3.6-flash__gaa.py    # Gemini, Ga only
```

| Track | Knobs | Full override |
|---|---|---|
| Khaya | `LANGUAGE_CODE` — the `?language=` value | `transcribe(wav_bytes, khaya_code, key)` |
| Google ASR | `LANGUAGE_CODE` — BCP-47 code | `transcribe(pcm_bytes, sample_rate, google_code)` |
| Gemini | `PROMPT`, `LANGUAGE_NAME` | `transcribe(wav_bytes, prompt)` |

`PROMPT` is the per-language prompt Gemini actually receives — tune the wording,
orthography hints or examples for one language and the other languages are
unaffected. `LANGUAGE_CODE = None` disables that language for that track.

Languages an API does not support yet ship a recipe with `LANGUAGE_CODE = None`
and a comment, so enabling one is a one-line pull request.

## How to update a recipe

1. Open your model's file in [`recipes/`](.) — e.g.
   [`KhayaAI_w2v-bert-ada.py`](./KhayaAI_w2v-bert-ada.py).
2. Edit the constants, or replace/extend `build_wrapper()` with a custom wrapper
   (subclass, custom `transcribe_batch`, LM post-processing, tokenizer handling,
   anything — it's plain Python).
3. Open a pull request on [GhanaNLP/nsanku-ASR](https://github.com/GhanaNLP/nsanku-ASR).
4. Once merged, the next benchmark run executes your code.

Every model row in the leaderboard links to its recipe via the **code** badge.

## Regenerating

```bash
python3 generate_recipes.py       # per-model recipes (HF models)
python3 generate_api_recipes.py   # per-language recipes (Khaya / Google / Gemini)
```

Both scripts only create recipes that don't exist yet.
**Existing recipe files are never overwritten** — once a recipe exists it is
treated as the author's code, so edits always survive regeneration.
