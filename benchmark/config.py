"""Configuration for nsanku-ASR benchmarking."""

import os
from pathlib import Path

# Load .env if present (for local dev)
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    with open(_env) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# Dataset — evaluation set with per-category configs ({category}_{Language}_{iso})
GHANA_SPEECH_EVAL = "ghananlpcommunity/ghana-speech-eval"
NUM_SAMPLES = 1000  # samples per category-config (dataset provides up to 1000)
SAMPLE_RATE = 16000

# HuggingFace authentication — set HF_TOKEN in .env or environment.
# Used for authenticated (gated/org) model + dataset access.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Benchmark models from both organizations and individual accounts.
ORG_ONLY = False

# Exclude generic "supports all languages" base models. A model is kept only if it
# explicitly targets a modest set of languages (HF language-tag count <= this).
MAX_LANG_TAGS = 60

# Only benchmark models trained for a SINGLE language (drops multilingual models such
# as Simba, Sunbird, Whisper/MMS bases, and multi-language fine-tunes). A model counts
# as single-language when its HF config language tags resolve to exactly one distinct
# language (iso 639-1/639-3 synonyms and Akan variants are collapsed).
SINGLE_LANGUAGE_ONLY = True

# When two eligible models share the same basename across different orgs, keep one:
# drop the ghananlpcommunity copy if present, else keep the earliest-published repo.
DEDUP_SAME_BASENAME = True

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
BENCHMARK_DIR = ROOT / "benchmarks"
TRANSCRIPTIONS_DIR = ROOT / "transcriptions"
LANG_CONFIG = ROOT / "languages" / "ghana_languages.yaml"
RESULTS_FILE = DATA_DIR / "ghana_asr_results.json"
EVAL_CONFIGS_FILE = DATA_DIR / "eval_configs.json"   # iso -> {language, categories:[{category, config}]}
OWNER_TYPES_FILE = DATA_DIR / "owner_types.json"     # cache: owner -> "org" | "user"

# Models to benchmark per language
# Loaded dynamically from ghana_asr_results.json at runtime
# Or override here for specific cases

# GPU batch config
# H200: 140GB VRAM (but leave room for other processes)
# Keep batch sizes modest — ~60GB max usage
BATCH_SIZE = {
    "tiny": 32,     # < 100M params
    "small": 16,    # 100M-500M params
    "medium": 8,    # 500M-1B params
    "large": 4,     # 1B-3B params
    "xlarge": 1,    # >3B params
}

# Compute dtype
TORCH_DTYPE = "bfloat16"  # bf16 recommended for H200 (BF16 + TF32)
