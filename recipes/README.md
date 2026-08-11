# Model recipes

One file per benchmarked model: `recipes/{owner}_{model}.md`. Each recipe records
the **exact inference code used** to evaluate that model on this benchmark,
along with the architecture and precision.

## Why recipes exist

Some models need special inference: custom tokenizers, language decoding, LM
post-processing, or non-standard loaders. If a model was benchmarked with the
wrong code its WER is not representative. These recipes let model authors see
exactly how their model was run, and correct it.

## How to update a recipe

1. Find your model's recipe in [`recipes/`](.) — e.g.
   [`KhayaAI_w2v-bert-ada.md`](./KhayaAI_w2v-bert-ada.md).
2. Edit the `Inference code used` section (or the notes) with the correct code.
3. Open a pull request on [GhanaNLP/nsanku-ASR](https://github.com/GhanaNLP/nsanku-ASR).
4. Once merged, the benchmark will use your updated recipe on the next run.

Every model row in the leaderboard links to its recipe.

## Regenerating

Recipes are generated from the real evaluation code so they never drift:

```bash
python3 generate_recipes.py
```

The embedded code is pulled directly from `benchmark/models.py` (Whisper / CTC
wrappers) and `benchmark/khaya.py` (hosted API) via `inspect.getsource`, so the
recipe always shows what actually ran.
