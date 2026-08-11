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
python3 generate_recipes.py
```

`generate_recipes.py` creates recipes for models that don't have one yet.
**Existing recipe files are never overwritten** — once a recipe exists it is
treated as the author's code, so edits always survive regeneration.
