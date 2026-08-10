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
    SINGLE_LANGUAGE_ONLY, DEDUP_SAME_BASENAME, LANG_CONFIG,
)

LANGTAG_COUNTS_FILE = DATA_DIR / "model_lang_tags.json"  # cache: model_id -> [lang codes]
MODEL_CREATED_FILE = DATA_DIR / "model_created.json"     # cache: model_id -> ISO created date

# Collapse language-code synonyms to a single canonical language when counting how many
# distinct languages a model targets. iso 639-1 <-> 639-3 pairs are read from the language
# metadata; Akan variants (Twi/Akuapem/Asante) collapse together.
_ISO_SYNONYMS = None


def _canon_map():
    global _ISO_SYNONYMS
    if _ISO_SYNONYMS is not None:
        return _ISO_SYNONYMS
    import yaml
    m = {}
    try:
        meta = yaml.safe_load(open(LANG_CONFIG))
        for l in meta.get("languages", []):
            iso3 = l.get("iso_639_3"); iso1 = l.get("iso_639_1")
            if iso3:
                m.setdefault(iso3, iso3)
                if iso1:
                    m[iso1] = iso3
    except Exception:
        pass
    # Akan variants (Twi/Akuapem/Asante) collapse to one language — applied LAST so
    # they win over any iso 639-1/639-3 default (e.g. tw -> twi) from the metadata.
    for t in ("ak", "aka", "twi", "tw", "akan", "atw", "ak-gh"):
        m[t] = "aka"
    _ISO_SYNONYMS = m
    return m


def canon_lang(tag):
    return _canon_map().get(tag.lower(), tag.lower())


def distinct_language_count(model_id, cache=None):
    """Number of distinct languages a model's config targets (synonyms collapsed)."""
    tags = lang_tags(model_id, cache)
    return len({canon_lang(t) for t in tags})

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
        if SINGLE_LANGUAGE_ONLY and distinct_language_count(name, ltags) > 1:
            continue
        if iso_codes is not None and not targets_language(name, iso_codes, ltags):
            continue
        kept.append(m)
    if DEDUP_SAME_BASENAME:
        kept = dedupe_by_basename(kept)
    return kept


def _load_created_cache():
    if MODEL_CREATED_FILE.exists():
        try:
            return json.load(open(MODEL_CREATED_FILE))
        except Exception:
            return {}
    return {}


def model_created_at(model_id, cache=None):
    """ISO creation timestamp of a model repo (cached); '' on failure."""
    cache = _load_created_cache() if cache is None else cache
    if model_id in cache:
        return cache[model_id]
    try:
        r = requests.get(f"https://huggingface.co/api/models/{model_id}",
                         headers=_headers(), timeout=15)
        created = r.json().get("createdAt", "") if r.status_code == 200 else ""
    except Exception:
        created = ""
    cache[model_id] = created
    MODEL_CREATED_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cache, open(MODEL_CREATED_FILE, "w"), indent=2, sort_keys=True)
    return created


def dedupe_by_basename(models):
    """Collapse models that share a basename across different orgs.

    Rule: if one copy is owned by ghananlpcommunity, drop that copy (keep the other);
    otherwise keep the earliest-published repo.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for m in models:
        groups[m["name"].split("/")[-1]].append(m)
    kept = []
    for base, group in groups.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        non_gnlp = [m for m in group if m["name"].split("/")[0] != "ghananlpcommunity"]
        pool = non_gnlp if non_gnlp else group
        if len(pool) == 1:
            kept.append(pool[0])
        else:
            kept.append(min(pool, key=lambda m: model_created_at(m["name"]) or "9999"))
    # preserve original order
    order = {id(m): i for i, m in enumerate(models)}
    return sorted(kept, key=lambda m: order[id(m)])
