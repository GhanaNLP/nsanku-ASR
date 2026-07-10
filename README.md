# nsanku-ASR

Benchmarking ASR models on Ghanaian languages — WER/CER evaluation across 41 Ghanaian language varieties using the [ghana-speech](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech) dataset.

## Languages

42 Ghanaian language varieties across 41 unique ISO 639-3 codes (Twi Akuapem/Twi Asante share `twi`).

| Language | ISO | Utterances | Hours | ASR Models on HF |
|---|---|---|---|---|
| Akuapem_Twi / Asante_Twi | twi | 196,033 | 263.27h | 46 |
| Hausa | hau | 92,082 | 152.77h | 43 |
| Ewe | ewe | 108,240 | 160.87h | 30 |
| Kabiye | kbp | 12,302 | 17.63h | 5 |
| Dagbani | dag | 46,526 | 73.06h | 4 |
| Dagaare | dga | 7,369 | 15.14h | 4 |
| Fante | fat | 60,933 | 122.03h | 2 |
| Nkonya | nko | 13,878 | 18.42h | 2 |
| +33 more | | | | 1 each |

Total HF ASR models found: **176**

## Pipeline

1. **Gemini 3.1 Flash Lite** — all 41 languages (API-based, no GPU needed)
2. **HuggingFace ASR models** — 8 languages with dedicated models (GPU required)

### Results

- `benchmarks/{iso}.yaml` — WER/CER scores per model, sorted best-to-worst
- `transcriptions/{iso}_{model}.csv` — Per-sample reference vs hypothesis pairs with sample-level WER/CER
- `model_status.csv` — Status table (pass/fail) for every model × language combination

## Structure

```
nsanku-ASR/
├── benchmark/                   # Benchmarking pipeline
│   ├── config.py                # Config: paths, batch sizes, language mapping, HF_TOKEN
│   ├── dataset.py               # Ghana Speech dataset loader (streaming, decode=False + soundfile)
│   ├── models.py                # ASR model wrappers (Whisper seq2seq, wav2vec2/MMS CTC)
│   ├── gemini.py                # Gemini 3.1 Flash Lite evaluation
│   ├── metrics.py               # WER / CER computation
│   └── evaluate.py              # Orchestrator: load model → transcribe → metrics → save
├── benchmarks/                  # Per-language YAML results
├── transcriptions/             # Per-sample reference/hypothesis CSVs
├── data/                        # HF search results, language metadata
├── languages/                   # Language metadata + dataset stats
├── scripts/                     # HF scraper, requirements
├── pipeline.py                  # Master runner: Gemini → HF models
├── run_benchmark.py             # CLI: run HF model benchmarks
├── run_gemini.sh                # Shell wrapper for Gemini evaluation
├── run_all.sh                   # Full pipeline script
├── generate_model_status.py     # Generate model_status.csv
├── results_summary.py           # Print benchmark results table
├── search_asr.py                # Search HF for ASR models
├── .env.example                 # Template for API keys
└── README.md
```

## Setup

```bash
# Create virtualenv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt

# Copy and fill in environment variables
cp .env.example .env
# Edit .env: set GEMINI_API_KEY and HF_TOKEN
```

### Environment Variables

| Variable | Purpose | Required for |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API access | Gemini evaluation |
| `HF_TOKEN` | HuggingFace authenticated access (gated models) | HF model evaluation |

## Usage

### Full pipeline (Gemini + HF models)

```bash
# On H200 GPU — runs Gemini on all 41 langs, then HF models on 8 langs
python3 -u pipeline.py 2>&1 | tee /tmp/nsanku_pipeline.log
```

### Gemini only

```bash
./run_gemini.sh twi ewe dag

# Or via Python
python3 -c "from benchmark.gemini import evaluate_gemini; evaluate_gemini('twi')"
```

### HF models only

```bash
# All languages with models
python run_benchmark.py

# Specific languages
python run_benchmark.py --langs twi ewe dag

# Filter by model name
python run_benchmark.py --langs twi --models whisper

# Dry run (preview, no GPU needed)
python run_benchmark.py --dry-run
```

### View results

```bash
python results_summary.py           # Summary table
python generate_model_status.py     # model_status.csv (pass/fail per model)
```

## Pipeline Details

### Dataset
- Source: `ghananlpcommunity/ghana-speech` (42 configs)
- 300 samples per language (configurable in `benchmark/config.py`)
- Audio decoded with `soundfile` (avoids torchcodec CUDA dependency)
- Automatic resampling to 16kHz
- Fields: `id`, `language`, `text` (reference transcript), `duration`, `audio`

### Models
- **Whisper-based** (seq2seq): `transformers` AutoModelForSpeechSeq2Seq
- **CTC-based** (wav2vec2/MMS/HuBERT): `transformers` AutoModelForCTC
- **Gemini 3.1 Flash Lite**: Google GenAI API with base64-encoded WAV audio

### Metrics
- WER (Word Error Rate): tokenized on whitespace
- CER (Character Error Rate): character-level
- Both normalized (uppercase, strip punctuation)

### GPU Config (H200)
- 140 GB VRAM, CUDA 12.8, compute capability 9.0 (Hopper)
- BF16 + TF32 compute
- cuDNN disabled (`torch.backends.cudnn.enabled = False`) due to Hopper compatibility issue
- `attn_implementation="eager"` for all models
- HF cache on large volume: `HF_HOME=/mnt/volume_d2wey28/hf_cache`

## Results Format

### benchmarks/{iso}.yaml
```yaml
iso_639_3: twi
num_samples: 300
benchmarks:
  - model: GiftMark/akan-whisper-model
    model_url: https://huggingface.co/GiftMark/akan-whisper-model
    params: 0.2B
    wer: 0.1224
    cer: 0.0393
    avg_seconds_per_sample: 0.15
    source: evaluated
  - model: google/gemini-3.1-flash-lite
    model_url: https://ai.google.dev/models/gemini-3.1-flash-lite
    params: API
    wer: 0.1761
    cer: 0.0410
    source: evaluated
```

### transcriptions/{iso}_{model}.csv
```csv
sample_id,reference,hypothesis,wer,cer
0,BERESOSƐM 1.,[beresosɛm 1],0.0,0.0
1,Efi Adam So Kosi Noa Mma So.,[efi adam so kosi noa mma so],0.0,0.0
```

### model_status.csv
```csv
model,language,params,wer,cer,status,fail_reason
GiftMark/akan-whisper-model,twi,0.2B,0.1224,0.0393,pass,
Kennethdot/kasanoma_whisper,twi,?,,,fail,gated_repo
```