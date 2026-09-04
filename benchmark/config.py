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

# Only benchmark models published by organizations (drop personal accounts).
ORG_ONLY = True

# Namespaces to treat as organizations even though HuggingFace classifies them
# as personal accounts (e.g. FarmerlineML). Ensures their models are not dropped
# by the org-only rule, are included in org scans, and are exempt from the
# model-card gate below.
#
# GhanaNLP was here while khaya-asr-v3 sat under that namespace; the model is
# KhayaAI's, and KhayaAI is a real HF org, so the override is no longer needed.
ORG_OVERRIDES = {"FarmerlineML", "Qlerqly", "katrintomanek"}

# A model only reaches the eval list if it ships a real (non-placeholder) model
# card. Licenses are NOT required — too many otherwise-good org models on HF
# (cdli, FarmerlineML, ...) simply never declare one.
# Namespaces in ORG_OVERRIDES are exempt from this gate.
REQUIRE_MODEL_CARD = True

# Minimum length of meaningful model-card prose (front-matter, headings and HTML
# comments stripped) for a card to count as real.
MIN_CARD_CHARS = 150

# Exclude generic "supports all languages" base models. A model is kept only if it
# explicitly targets a modest set of languages (HF language-tag count <= this).
MAX_LANG_TAGS = 60

# Models excluded by decision rather than by rule. Each entry needs a reason:
# the point is to skip work we know is wasted, not to hide results.
SKIP_MODELS = {
    # Runs correctly but costs ~2.5s/sample even through ONNX (the PyTorch path
    # was ~8s), i.e. hours per language for one model. Judged not worth the
    # budget on 2026-08-12; the ONNX export and recipe are kept, so re-enabling
    # is a one-line change.
    "ghananlpcommunity/qwen3-asr-akuapem-twi",
    # Repo ships no feature-extractor config, so it cannot load at all. Failed
    # every attempt across three runs; retrying it just burns a download.
    "ghananlpcommunity/FarmerlineWav2Vec-Twi-Demo",
}

# Hard ceiling on model size. Anything larger is not benchmarked at all — the
# eval budget does not stretch to multi-billion-parameter models generating a
# transcript token by token (a 7B audio LLM is ~20x slower per sample than a
# w2v-bert CTC model, before counting the ~16GB download).
MAX_PARAMS = 2_500_000_000  # 2.5B — headroom so a ~2B fine-tune is not lost on a rounding edge

# Only benchmark models trained for a SINGLE language (drops multilingual models such
# as Simba, Sunbird, Whisper/MMS bases, and multi-language fine-tunes). A model counts
# as single-language when its HF config language tags resolve to exactly one distinct
# language (iso 639-1/639-3 synonyms and Akan variants are collapsed).
SINGLE_LANGUAGE_ONLY = True

# Multilingual models included by decision despite SINGLE_LANGUAGE_ONLY. Each still only
# runs on the languages it explicitly declares, and still must pass the org/ASR/card
# gates — it is only exempt from the single-language, general-base (MAX_LANG_TAGS) and
# size (MAX_PARAMS) rules. Use for notable multilingual baselines we deliberately want
# on the board (e.g. Sunbird's 51-African-language Whisper).
FORCE_INCLUDE_MODELS = {
    "Sunbird/asr-whisper-51-african-languages",
    # Personal-account model (Qlerqly is in ORG_OVERRIDES) the user explicitly
    # asked to benchmark: multilingual Ghanaian (Akan, Dagbani, Ewe, Ga) so it
    # needs the single-language gate lifted, like Sunbird.
    "Qlerqly/griot-nano-1",
    # Personal-account (katrintomanek is in ORG_OVERRIDES) Whisper Akan/Twi
    # fine-tunes the user explicitly asked to benchmark. They declare no HF
    # language tags, so without FORCE_INCLUDE the name-token Twi/Akan match
    # alone would be the only evidence; forcing keeps them in the pool.
    "katrintomanek/whisper-large-v3-turbo_Akan_standardspeech_specaugment",
    "katrintomanek/whisper-small_twi-standardspeech_v2",
    # FP16 CTranslate2 (faster-whisper) re-export of the katrintomanek large-v3
    # turbo weights, published under the GhanaNLP org. The ct2 format is not
    # loadable by the generic Whisper wrapper, so it needs its own recipe.
    "ghananlpcommunity/whisper-large-v3-turbo-twi-ctranslate",
}

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
BENCHMARK_DIR = ROOT / "benchmarks"
TRANSCRIPTIONS_DIR = ROOT / "transcriptions"
LANG_CONFIG = ROOT / "languages" / "ghana_languages.yaml"
RESULTS_FILE = DATA_DIR / "ghana_asr_results.json"
EVAL_CONFIGS_FILE = DATA_DIR / "eval_configs.json"   # iso -> {language, categories:[{category, config}]}
OWNER_TYPES_FILE = DATA_DIR / "owner_types.json"     # cache: owner -> "org" | "user"
MODEL_LICENSES_FILE = DATA_DIR / "model_licenses.json"  # cache: model_id -> license string ("" = none)
MODEL_CARDS_FILE = DATA_DIR / "model_cards.json"        # cache: model_id -> reason ("" = card ok)
MODEL_PARAMS_FILE = DATA_DIR / "model_params.json"      # cache: model_id -> param count (0 = unknown)

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
