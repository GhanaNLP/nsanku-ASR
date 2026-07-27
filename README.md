# nsanku-ASR

Benchmarking **organization-owned** ASR models on Ghanaian languages — WER/CER evaluation
across 43 Ghanaian language varieties using the
[ghana-speech-eval](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech-eval) dataset.

## What's benchmarked

- **Only models owned by organizations** (not personal HuggingFace accounts). Owner type is
  detected via the HuggingFace API and cached in `data/owner_types.json`.
- **Authenticated access** — models and the eval dataset are loaded with an `HF_TOKEN`, so
  gated / org-restricted repos are reachable.
- Obvious **non-ASR** org models are excluded (TTS, forced-aligners, spoken-language-ID).

## Per-category scoring

The eval dataset splits each language into **categories** — `bible`, `jw`, `finance`,
`unicef` — as separate configs (`{category}_{Language}_{iso}`). A language appears in one or
more categories.

Each model is scored on **every category its language appears in**, and the reported
**WER/CER is the average across those categories**. The mapping of language → categories lives
in `data/eval_configs.json`.

Example: Twi has all four categories, so a model's final WER for Twi is the mean of its Bible,
JW, Finance, and UNICEF WERs.

## Languages & models

- **43 languages** in the eval set (adds Ga `gaa` and Ahanta `aha` beyond the original 41).
- Model discovery scrapes HuggingFace for ASR models tagged with each language, then filters
  to org-owned ASR models. Every language has at least one org model (multilingual models such
  as Whisper, MMS and Wav2Vec2-BERT cover the long tail).

## Pipeline

1. **Discover** — `search_asr.py` scrapes HF for ASR models per language → `data/ghana_asr_results.json`
2. **Filter** — `benchmark/owners.py` keeps only org-owned ASR models
3. **Evaluate** — for each language, each model is scored per category and averaged

### Results

- `benchmarks/{iso}.yaml` — averaged WER/CER per model with a `per_category` breakdown, sorted best-to-worst
- `transcriptions/{iso}_{category}_{model}.csv` — per-sample reference vs hypothesis with sample-level WER/CER

## Structure

```
nsanku-ASR/
├── benchmark/
│   ├── config.py       # Paths, dataset name, HF_TOKEN, ORG_ONLY, sample count
│   ├── dataset.py      # ghana-speech-eval loader (streaming, decode=False + soundfile)
│   ├── owners.py       # Org-vs-personal detection + org/non-ASR filtering (cached)
│   ├── models.py       # ASR wrappers (Whisper seq2seq bf16; CTC/MMS/w2v-bert fp32)
│   ├── metrics.py      # WER / CER (Levenshtein)
│   └── evaluate.py     # Per-category scoring → averaged final WER/CER
├── benchmarks/         # Per-language YAML results (per-category + averaged)
├── transcriptions/     # Per-sample reference/hypothesis CSVs
├── data/
│   ├── ghana_asr_results.json  # HF model scrape
│   ├── eval_configs.json       # iso -> {language, categories:[{category, config}]}
│   └── owner_types.json        # owner -> "org" | "user" (cache)
├── languages/          # Language metadata
├── scripts/            # HF scraper + requirements
├── space/              # HuggingFace Space leaderboard (Gradio app.py + static index.html)
├── pipeline.py         # Runs the full benchmark over all eval languages
├── run_benchmark.py    # CLI for targeted runs
└── search_asr.py       # HF ASR model search
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt

cp .env.example .env
# Edit .env: set HF_TOKEN
```

### Environment Variables

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Authenticated HuggingFace access (gated/org models + eval dataset) |

## Usage

GPU evaluations run on the Ghana NLP H200.

```bash
# Full benchmark — all eval languages (GPU)
python3 -u pipeline.py 2>&1 | tee /tmp/nsanku_pipeline.log

# Specific languages
python run_benchmark.py --langs twi ewe dag

# Filter by model name
python run_benchmark.py --langs hau --models whisper

# Preview (no GPU): languages, categories, org model counts
python run_benchmark.py --dry-run
```

## Pipeline details

### Dataset
- Source: `ghananlpcommunity/ghana-speech-eval` (57 category-configs across 43 languages)
- Up to 1000 samples per category-config (configurable via `NUM_SAMPLES` in `benchmark/config.py`)
- Audio decoded with `soundfile` (avoids torchcodec), resampled to 16 kHz
- Fields: `audio`, `text`, `language`, `iso`, `country`, `length`, `subset`

### Models
- **Whisper-based** (seq2seq): `AutoModelForSpeechSeq2Seq`, bf16
- **CTC-based** (wav2vec2 / MMS / Wav2Vec2-BERT / HuBERT): `AutoModelForCTC`, **fp32**
  (raw-waveform conv encoders hit a broken cuDNN path in bf16 on Hopper)

### Metrics
- WER (Word Error Rate) — whitespace-tokenized Levenshtein
- CER (Character Error Rate) — character-level Levenshtein
- Both normalized (uppercase, strip punctuation)

### GPU config (H200)
- 140 GB VRAM, CUDA 12.8, compute capability 9.0 (Hopper)
- `torch.backends.cudnn.enabled = False`, `attn_implementation="eager"`
- HF cache on the large volume: `HF_HOME=/mnt/volume_d2wey28/hf_cache`

## Results format

### benchmarks/{iso}.yaml
```yaml
iso_639_3: dag
language: Dagbani
num_samples_per_category: 300
categories: [bible, unicef]
benchmarks:
  - model: Sunbird/asr-whisper-51-african-languages
    model_url: https://huggingface.co/Sunbird/asr-whisper-51-african-languages
    owner: Sunbird
    params: 2B
    wer: 0.6499        # average across categories
    cer: 0.3014
    per_category:
      bible:  {wer: 0.7082, cer: 0.3379, samples: 300, valid: 300}
      unicef: {wer: 0.5916, cer: 0.2650, samples: 195, valid: 195}
    source: evaluated
```

## Leaderboard

Interactive leaderboard (HuggingFace Space): reads `benchmarks/*.yaml` from this repo and shows
the global best-per-language table, per-language breakdowns with per-category WER, and model
status. See `space/`.
