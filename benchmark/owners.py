"""Classify HuggingFace model owners as organizations vs personal accounts,
and filter model lists to org-owned ASR models only.

Org detection uses HuggingFace's public API:
  GET https://huggingface.co/api/organizations/{name}/overview  -> 200 for orgs
Results are cached to data/owner_types.json to avoid repeat lookups.
"""

import json

import requests

from .config import (
    OWNER_TYPES_FILE, HF_TOKEN, ORG_ONLY, MAX_LANG_TAGS, DATA_DIR,
)

LANGTAG_COUNTS_FILE = DATA_DIR / "model_lang_tags.json"  # cache: model_id -> [lang codes]

# Keywords marking a model as NOT automatic-speech-recognition.
# These org models exist but cannot transcribe (TTS, aligners, language-ID).
NON_ASR_KEYWORDS = [
    "tts", "text-to-speech",
    "forced-aligner", "forced_aligner", "aligner",
    "-slid", "slid-", "-lid-", "langid", "language-id", "spoken-language-id",
]


def _headers():
    return {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def _load_cache():
    if OWNER_TYPES_FILE.exists():
        try:
            return json.load(open(OWNER_TYPES_FILE))
        except Exception:
            return {}
    return {}


def _save_cache(cache):
    OWNER_TYPES_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cache, open(OWNER_TYPES_FILE, "w"), indent=2, sort_keys=True)


def owner_type(owner, cache=None):
    """Return 'org' or 'user' for a HuggingFace namespace, cached."""
    cache = _load_cache() if cache is None else cache
    if owner in cache:
        return cache[owner]
    try:
        r = requests.get(
            f"https://huggingface.co/api/organizations/{owner}/overview",
            headers=_headers(), timeout=15,
        )
        kind = "org" if r.status_code == 200 else "user"
    except Exception:
        kind = "user"
    cache[owner] = kind
    _save_cache(cache)
    return kind


def is_org(owner, cache=None):
    return owner_type(owner, cache) == "org"


def is_non_asr(model_id):
    m = model_id.lower()
    return any(kw in m for kw in NON_ASR_KEYWORDS)


def _load_langtag_cache():
    if LANGTAG_COUNTS_FILE.exists():
        try:
            return json.load(open(LANGTAG_COUNTS_FILE))
        except Exception:
            return {}
    return {}


def _save_langtag_cache(cache):
    LANGTAG_COUNTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cache, open(LANGTAG_COUNTS_FILE, "w"), indent=2, sort_keys=True)


def lang_tags(model_id, cache=None):
    """List of language codes a model's HF config explicitly declares (cached, lowercased)."""
    cache = _load_langtag_cache() if cache is None else cache
    if model_id in cache:
        return cache[model_id]
    try:
        r = requests.get(
            f"https://huggingface.co/api/models/{model_id}?full=true",
            headers=_headers(), timeout=20,
        )
        cd = (r.json().get("cardData") or {}) if r.status_code == 200 else {}
        langs = cd.get("language") or cd.get("languages") or []
        if isinstance(langs, str):
            langs = [langs]
        langs = [str(x).lower() for x in langs]
    except Exception:
        langs = []  # unknown -> empty (treated as not targeting any language)
    cache[model_id] = langs
    _save_langtag_cache(cache)
    return langs


def lang_tag_count(model_id, cache=None):
    return len(lang_tags(model_id, cache))


def is_general_model(model_id, cache=None):
    """True if the model is a generic 'all languages' base model (too many lang tags)."""
    return lang_tag_count(model_id, cache) > MAX_LANG_TAGS


def targets_language(model_id, iso_codes, cache=None):
    """True if the model's config explicitly declares one of the given codes.

    iso_codes: iterable of acceptable tokens for the language (639-3 and 639-1).
    Macrolanguage codes (e.g. 'ak' for Akan) are intentionally NOT auto-expanded —
    a model must name the specific language.
    """
    tags = set(lang_tags(model_id, cache))
    return bool(tags & set(iso_codes))


def filter_models(models, iso_codes=None):
    """Keep only org-owned ASR models that explicitly target the language.

    Drops: personal accounts, non-ASR models (TTS/aligner/LID), generic global
    base models (>MAX_LANG_TAGS languages), and — when `iso_codes` is given —
    any model whose config does not explicitly declare the language.

    `models` is a list of dicts each with a 'name' key ("owner/model").
    """
    otype = _load_cache()
    ltags = _load_langtag_cache()
    kept = []
    for m in models:
        name = m["name"]
        owner = name.split("/")[0]
        if is_non_asr(name):
            continue
        if ORG_ONLY and not is_org(owner, otype):
            continue
        if is_general_model(name, ltags):
            continue
        if iso_codes is not None and not targets_language(name, iso_codes, ltags):
            continue
        kept.append(m)
    return kept
