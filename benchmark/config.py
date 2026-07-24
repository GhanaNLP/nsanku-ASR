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
# Legacy dataset (kept for reference)
GHANA_SPEECH = "ghananlpcommunity/ghana-speech"
NUM_SAMPLES = 300  # samples per category-config
SAMPLE_RATE = 16000

# HuggingFace authentication — set HF_TOKEN in .env or environment.
# Used for authenticated (gated/org) model + dataset access.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Only benchmark models owned by organizations (not personal accounts).
ORG_ONLY = True

# Exclude generic "supports all languages" base models. A model is kept only if it
# explicitly targets a modest set of languages (HF language-tag count <= this).
# This drops global base models (Whisper ~99, MMS ~158, SeamlessM4T ~96, XEUS ~4024,
# faster-whisper ~100) while keeping language-specific and African-multilingual models
# (e.g. UBC-NLP Simba = 39 African langs, Sunbird = 51 African langs).
MAX_LANG_TAGS = 60

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
BENCHMARK_DIR = ROOT / "benchmarks"
TRANSCRIPTIONS_DIR = ROOT / "transcriptions"
LANG_CONFIG = ROOT / "languages" / "ghana_languages.yaml"
RESULTS_FILE = DATA_DIR / "ghana_asr_results.json"
EVAL_CONFIGS_FILE = DATA_DIR / "eval_configs.json"   # iso -> {language, categories:[{category, config}]}
OWNER_TYPES_FILE = DATA_DIR / "owner_types.json"     # cache: owner -> "org" | "user"

# Language → dataset config mapping
# The dataset uses "{LanguageName}_{iso}" naming
# We derive config name from the language metadata
LANG_TO_CONFIG = {
    "twi": "Akuapem_Twi_twi",   # Akuapem_Twi; Asante_Twi is separate but same iso
    "any": "Anyin_any",
    "avn": "Avatime_avn",
    "bud": "Bassar_Ntcham_bud",
    "bim": "Bimoba_bim",
    "biv": "Birifor_Southern_biv",
    "bib": "Bissa_bib",
    "bwu": "Buli_bwu",
    "ncu": "Chumburung_ncu",
    "dga": "Dagaare_dga",
    "dag": "Dagbani_dag",
    "ada": "Dangme_ada",
    "mzw": "Deg_mzw",
    "ewe": "Ewe_ewe",
    "fat": "Fante_fat",
    "ffm": "Fulfulde_Maasina_ffm",
    "acd": "Gikyode_acd",
    "gjn": "Gonja_gjn",
    "hau": "Hausa_hau",
    "kbp": "Kabiye_kbp",
    "xsm": "Kasem_xsm",
    "xon": "Konkomba_xon",
    "kma": "Konni_kma",
    "kus": "Kusaal_kus",
    "lef": "Lelemi_lef",
    "maw": "Mampruli_maw",
    "naw": "Nawuri_naw",
    "gur": "Ninkare_gur",
    "nko": "Nkonya_nko",
    "ntr": "Ntrubo_ntr",
    "nzi": "Nzema_nzi",
    "sig": "Paasaal_sig",
    "sfw": "Sehwi_sfw",
    "lip": "Sekpele_lip",
    "snw": "Selee_snw",
    "sil": "Sisaala_Tumulung_sil",
    "akp": "Siwu_akp",
    "tpm": "Tampulma_tpm",
    "kdh": "Tem_kdh",
    "bov": "Tuwuli_bov",
    "vag": "Vagla_vag",
}

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
