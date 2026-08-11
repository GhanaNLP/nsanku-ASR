# Model recipes

One file per benchmarked model: `recipes/{owner}_{model}.md`. Each recipe records
the **exact inference code used** to evaluate that model on this benchmark,
along with the architecture, precision, and the YAML front-matter overrides that
the benchmark will apply.

## Why recipes exist

Some models need special inference: custom tokenizers, language decoding, LM
post-processing, or non-standard loaders. If a model was benchmarked with the
wrong code its WER is not representative. These recipes let model authors see
exactly how their model was run, and correct it.

## How a recipe is used

Before evaluating a model, `benchmark/evaluate.py` reads its recipe's YAML
front-matter (`benchmark/recipes.py`) and passes any overrides into the model
wrapper. Supported overrides:

| Key | Applies to | Values |
|---|---|---|
| `language` | Whisper-style seq2seq | source language code, e.g. `twi` (default: auto) |
| `task` | Whisper-style seq2seq | `transcribe` or `translate` |
| `initial_prompt` | Whisper-style seq2seq | decoder prompt text |
| `ctc_decoder` | wav2vec2 / MMS / Wav2Vec2-BERT | `greedy` (only option today) |

So editing a recipe and opening a pull request **really changes the next run**.

## How to update a recipe

1. Find your model's recipe in [`recipes/`](.) — e.g.
   [`KhayaAI_w2v-bert-ada.md`](./KhayaAI_w2v-bert-ada.md).
2. Edit the YAML front-matter (the `---` block at the top) to set overrides, or
   add notes explaining custom code.
3. Open a pull request on [GhanaNLP/nsanku-ASR](https://github.com/GhanaNLP/nsanku-ASR).
4. Once merged, the next benchmark run applies your overrides.

Every model row in the leaderboard links to its recipe.

## Regenerating

Recipes are generated from the real evaluation code so they never drift:

```bash
python3 generate_recipes.py
```

The embedded code is pulled directly from `benchmark/models.py` (Whisper / CTC
wrappers) and `benchmark/khaya.py` (hosted API) via `inspect.getsource`, so the
recipe always shows what actually ran. Regenerating refreshes the embedded code
and metadata while **preserving any front-matter values you've set**.
