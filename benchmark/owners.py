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
    SINGLE_LANGUAGE_ONLY, LANG_CONFIG, ORG_OVERRIDES,
    MODEL_LICENSES_FILE, MODEL_CARDS_FILE, REQUIRE_MODEL_CARD, MIN_CARD_CHARS,
    MODEL_PARAMS_FILE, MAX_PARAMS,
)

LANGTAG_COUNTS_FILE = DATA_DIR / "model_lang_tags.json"  # cache: model_id -> [lang codes]

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
    "tts", "text-to-speech", "vits", "voxcpm",
    "forced-aligner", "forced_aligner", "aligner",
    "-slid", "slid-", "-lid-", "langid", "language-id", "spoken-language-id",
]


# Repos that ARE ASR but cannot be loaded by benchmark/models.py: language models
# used for decoding rather than transcription, and runtime-specific exports that
# need a different engine (CTranslate2/faster-whisper, ONNX Runtime, llama.cpp).
# Benchmarking them only produces null rows — ghananlpcommunity/…_farmerline-ct2
# is already in the leaderboard with wer: null for exactly this reason.
UNSUPPORTED_ARTIFACT_TOKENS = {
    "kenlm", "arpa", "ngram", "2gram", "3gram", "4gram", "5gram", "6gram",
    "ct2", "ctranslate2", "onnx", "gguf", "openvino", "tflite", "coreml",
    "int4", "int8",
}


def is_unsupported_artifact(model_id):
    """True if the repo is an LM or a runtime export we cannot run as ASR."""
    import re
    tokens = set(re.split(r"[^a-z0-9]+", model_id.split("/")[-1].lower()))
    return bool(tokens & UNSUPPORTED_ARTIFACT_TOKENS)


def model_params(model_id, cache=None):
    """Parameter count from the repo's safetensors index; 0 when unknown. Cached."""
    own = cache is None
    cache = _load_json_cache(MODEL_PARAMS_FILE) if own else cache
    if model_id in cache:
        return cache[model_id]
    total = 0
    try:
        r = requests.get(f"https://huggingface.co/api/models/{model_id}",
                         headers=_headers(), timeout=20)
        if r.status_code == 200:
            st = r.json().get("safetensors") or {}
            total = int(st.get("total") or 0)
            if not total:
                # Older repos expose only a per-dtype breakdown.
                total = int(sum((st.get("parameters") or {}).values()) or 0)
    except Exception:
        total = 0
    cache[model_id] = total
    _save_json_cache(MODEL_PARAMS_FILE, cache)
    return total


def is_too_large(model_id, cache=None):
    """True if the model exceeds MAX_PARAMS. Unknown size counts as small.

    A repo with no safetensors metadata cannot be measured without downloading
    it, and most such repos are old small CTC checkpoints — excluding them on
    suspicion would silently drop real models.
    """
    n = model_params(model_id, cache)
    return bool(n) and n > MAX_PARAMS


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
    """Return 'org' or 'user' for a HuggingFace namespace, cached.

    Namespaces listed in ORG_OVERRIDES are always treated as orgs regardless of
    what HuggingFace's org lookup returns.
    """
    if owner in ORG_OVERRIDES:
        return "org"
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


def targets_language(model_id, iso_codes, cache=None, name_tokens=()):
    """True if the model explicitly declares/names one of the given codes.

    iso_codes: acceptable ISO tokens for the language (639-3, 639-1, aliases).
    Macrolanguage codes (e.g. 'ak' for Akan) are intentionally NOT auto-expanded —
    a model must name the specific language.

    Declared HF language tags are matched against `iso_codes` ONLY, because tags
    are ISO codes: a tag of "tem" means Temne (tem), never Tem/Kotokoli (kdh),
    whose English name merely happens to be spelled the same.

    `name_tokens` (the language's English name, e.g. 'hausa') is used only for
    models that declare NO tags, where the model's own basename is the sole
    evidence — "w2v-bert-2.0-hausa_250_250h" or "w2v-bert-2.0_twi_alpha_v1". A
    token must match whole (delimited by non-alphanumerics) so names like
    "brianyan918" don't match "any"; 1-2 letter codes are too noisy to use.
    """
    tags = set(lang_tags(model_id, cache))
    if tags & set(iso_codes):
        return True
    if not tags:
        import re
        basename = set(re.split(r"[^a-z0-9]+", model_id.split("/")[-1].lower()))
        wanted = {t for t in set(iso_codes) | set(name_tokens) if len(t) >= 3}
        return bool(basename & wanted)
    return False


# ---------------------------------------------------------------------------
# Model-card quality (a declared license is reported but NOT required)
# ---------------------------------------------------------------------------

# License values that count as "no license declared" (reporting only).
NO_LICENSE = {"", "none", "unknown", "no-license", "no license", "???"}

# Substrings marking a README as an auto-generated / unfilled stub rather than a
# real model card.
CARD_PLACEHOLDER_MARKERS = [
    "provide a quick summary of what the model is/does",
    "provide a longer summary of what this model is",
    "this model card has been automatically generated",
    "this model card is under construction",
    "this model has no card",
    "no model card",
    "write a model card here",
    "more information needed",
    "lorem ipsum",
]


def _load_json_cache(path):
    if path.exists():
        try:
            return json.load(open(path))
        except Exception:
            return {}
    return {}


def _save_json_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cache, open(path, "w"), indent=2, sort_keys=True)


def model_license(model_id, cache=None):
    """Declared license of a model ("" when none), cached."""
    own = cache is None
    cache = _load_json_cache(MODEL_LICENSES_FILE) if own else cache
    if model_id in cache:
        return cache[model_id]
    lic = ""
    try:
        r = requests.get(f"https://huggingface.co/api/models/{model_id}?full=true",
                         headers=_headers(), timeout=20)
        if r.status_code == 200:
            lic = str((r.json().get("cardData") or {}).get("license") or "")
    except Exception:
        lic = ""
    cache[model_id] = lic
    _save_json_cache(MODEL_LICENSES_FILE, cache)
    return lic


def license_ok(model_id, cache=None):
    return model_license(model_id, cache).strip().lower() not in NO_LICENSE


def _card_body(text):
    """README prose with YAML front-matter, HTML comments and headings removed."""
    import re
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2]
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"^\s*#+.*$", " ", text, flags=re.M)     # headings
    text = re.sub(r"^\s*[-*|>]+\s*", " ", text, flags=re.M)  # list/table/quote markers
    return re.sub(r"\s+", " ", text).strip()


def card_problem(model_id, cache=None):
    """'' if the model ships a real card, else a short reason string. Cached."""
    own = cache is None
    cache = _load_json_cache(MODEL_CARDS_FILE) if own else cache
    if model_id in cache:
        return cache[model_id]
    try:
        r = requests.get(f"https://huggingface.co/{model_id}/raw/main/README.md",
                         headers=_headers(), timeout=20)
        if r.status_code != 200:
            reason = f"no README (HTTP {r.status_code})"
        else:
            body = _card_body(r.text)
            low = body.lower()
            hit = next((k for k in CARD_PLACEHOLDER_MARKERS if k in low), None)
            if hit:
                reason = f"placeholder card ({hit!r})"
            elif len(body) < MIN_CARD_CHARS:
                reason = f"card too short ({len(body)} chars)"
            else:
                reason = ""
    except Exception as e:
        reason = f"card fetch failed ({type(e).__name__})"
    cache[model_id] = reason
    _save_json_cache(MODEL_CARDS_FILE, cache)
    return reason


def card_ok(model_id, cache=None):
    return card_problem(model_id, cache) == ""


def warm_caches_from_universe(models):
    """Seed the language-tag and license caches from org-scan `cardData`.

    Avoids one HF API call per model when a caller already has the metadata.
    `models` is a list of dicts with 'name' and optionally 'license'/'language'.
    Empty values are ignored rather than cached: caching "" or [] would mask a
    real license/tag behind a cache hit and no live lookup would ever correct it.
    """
    ltags = _load_langtag_cache()
    lic = _load_json_cache(MODEL_LICENSES_FILE)
    for m in models:
        name = m.get("name")
        if not name:
            continue
        langs = m.get("language") or []
        if isinstance(langs, str):
            langs = [langs]
        if langs and name not in ltags:
            ltags[name] = [str(x).lower() for x in langs]
        if m.get("license") and name not in lic:
            lic[name] = str(m["license"])
    _save_langtag_cache(ltags)
    _save_json_cache(MODEL_LICENSES_FILE, lic)
    return ltags, lic


def filter_models(models, iso_codes=None, name_tokens=(), require_card=None):
    """Keep only org-owned ASR models that explicitly target the language.

    Drops: personal accounts, non-ASR models (TTS/aligner/LID), LMs and runtime
    exports we cannot load (kenlm/ct2/onnx/gguf — see UNSUPPORTED_ARTIFACT_TOKENS),
    generic global
    base models (>MAX_LANG_TAGS languages), models above MAX_PARAMS parameters,
    models shipping only a placeholder
    model card, and — when `iso_codes` is given — any model whose config does
    not explicitly declare the language (see `targets_language` for how
    `name_tokens` is used). A declared license is NOT required.

    Namespaces in ORG_OVERRIDES bypass the model-card gate. Pass
    `require_card=False` to skip that gate entirely (used by the curation
    report to show why candidates fail).

    `models` is a list of dicts each with a 'name' key ("owner/model").
    """
    if require_card is None:
        require_card = REQUIRE_MODEL_CARD
    otype = _load_cache()
    ltags = _load_langtag_cache()
    cards = _load_json_cache(MODEL_CARDS_FILE)
    sizes = _load_json_cache(MODEL_PARAMS_FILE)
    kept = []
    for m in models:
        name = m["name"]
        owner = name.split("/")[0]
        if is_non_asr(name) or is_unsupported_artifact(name):
            continue
        if ORG_ONLY and not is_org(owner, otype):
            continue
        if is_general_model(name, ltags):
            continue
        if is_too_large(name, sizes):
            continue
        if SINGLE_LANGUAGE_ONLY and distinct_language_count(name, ltags) > 1:
            continue
        if iso_codes is not None and not targets_language(name, iso_codes, ltags,
                                                          name_tokens):
            continue
        if require_card and owner not in ORG_OVERRIDES and not card_ok(name, cards):
            continue
        kept.append(m)
    return kept
