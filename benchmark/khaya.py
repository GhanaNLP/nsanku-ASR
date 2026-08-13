"""Hosted-API ASR track — Khaya (GhanaNLP).

Transcribes the ghana-speech-eval audio with the Khaya ASR API
(POST https://translation-api.ghananlp.org/asr/v3/transcribe?language=<code>,
header Ocp-Apim-Subscription-Key, body audio/wav -> {"text": ...}).

Khaya is a hosted multilingual API and is included explicitly (exempt from the
single-language model filter). It is scored per category and averaged, like the
other tracks. Results go to benchmarks_api/{iso}.yaml and are merged into
benchmarks/ tagged model_class="non-llm".
"""

import os
import time
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yaml

from .config import NUM_SAMPLES, ROOT
from .dataset import load_eval_samples
from .evaluate import load_eval_configs, language_categories, save_transcriptions, _score
from .recipes import load_lang_recipe, recipe_get

# Load .env for KHAYA_API_KEY
_env = ROOT / ".env"
if _env.exists():
    for _line in open(_env):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

API_URL = "https://translation-api.ghananlp.org/asr/v3/transcribe"
MODEL_ID = "KhayaAI/khaya-asr-v3"
MODEL_URL = "https://translation-api.ghananlp.org/"
MAX_WORKERS = 8
MAX_RETRIES = 3

# eval iso_639_3 -> Khaya API language code (only Ghanaian languages Khaya supports)
EVAL_TO_KHAYA = {
    # Khaya exposes both dialects; with the eval set split by dialect they now
    # map one-to-one instead of one endpoint straddling mixed audio. Measured on
    # 60 samples/category: atw beat twi 8.93% vs 34.84% on the Akuapem-heavy
    # bible split, and lost 88.69% vs 49.52% on jw.
    "twi_akuapem": "atw",   # Akuapem Twi
    "twi_asante": "twi",    # Twi (Asante)
    "ada": "ada", "bwu": "bwu", "dag": "dag", "dga": "dga", "ewe": "ewe",
    "fat": "fat", "gaa": "gaa", "gjn": "gjn", "gur": "gur", "hau": "hau",
    "kus": "kus", "maw": "maw", "nzi": "nzi", "xsm": "xsm",
    "xon": "xon_likpakpaanl",   # Konkomba (main variant)
}

API_BENCHMARK_DIR = ROOT / "benchmarks_api"


def _encode_wav(audio_array, sample_rate=16000):
    import soundfile as sf
    buf = BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _transcribe(wav_bytes, khaya_code, key):
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "audio/wav"}
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(f"{API_URL}?language={khaya_code}", headers=headers,
                              data=wav_bytes, timeout=120)
            if r.status_code == 200:
                return (r.json().get("text") or "").strip()
            if r.status_code in (429, 500, 503) and attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1)); continue
            return ""
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
    return ""


def _has_result(iso_code):
    """True if THIS track already scored this language.

    Must match on MODEL_ID: benchmarks_api/{iso}.yaml is shared with the other
    hosted-API tracks, so "any benchmark has a wer" would make Khaya skip every
    language Google had already done, and vice versa.
    """
    path = API_BENCHMARK_DIR / f"{iso_code}.yaml"
    if not path.exists():
        return False
    d = yaml.safe_load(open(path)) or {}
    want = {c for c, _ in language_categories(iso_code)}
    for b in d.get("benchmarks", []):
        if b.get("model") != MODEL_ID or b.get("wer") is None:
            continue
        # Scored, but the reported WER averages whatever categories existed
        # then; a language that has since gained one must be re-run.
        return want <= set(b.get("per_category") or {})
    return False


def _save(iso_code, language, category_names, result):
    """Merge this track's result into benchmarks_api/{iso}.yaml.

    The file is shared with the other hosted-API tracks, so the existing entries
    are kept and only this model's row is replaced — writing [result] outright
    deleted whichever track had run first.
    """
    API_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = API_BENCHMARK_DIR / f"{iso_code}.yaml"
    existing = yaml.safe_load(open(path)) if path.exists() else None
    entries = (existing or {}).get("benchmarks", []) if existing else []
    merged = [b for b in entries if b.get("model") != result["model"]] + [result]
    cats = list(dict.fromkeys((existing or {}).get("categories", []) + category_names))
    out = {
        "iso_639_3": iso_code,
        "language": language,
        "num_samples_per_category": NUM_SAMPLES,
        "categories": cats,
        "benchmarks": merged,
    }
    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def evaluate_khaya(iso_code):
    """Evaluate the Khaya ASR API on one language across its eval categories."""
    key = os.environ.get("KHAYA_API_KEY")
    if not key:
        raise ValueError("KHAYA_API_KEY required (set in .env).")
    # Per-language recipe (recipes/KhayaAI_khaya-asr-v3__{iso}.py) owns the API
    # language code and may replace the transcribe call entirely.
    recipe = load_lang_recipe(MODEL_ID, iso_code)
    khaya_code = recipe_get(recipe, "LANGUAGE_CODE", EVAL_TO_KHAYA.get(iso_code))
    transcribe = recipe_get(recipe, "transcribe", _transcribe)
    if not khaya_code:
        print(f"  {iso_code}: not supported by Khaya API - skipping")
        return
    cats = language_categories(iso_code)
    if not cats:
        return
    if _has_result(iso_code):
        print(f"  Khaya already done for {iso_code} - skipping")
        return

    meta = load_eval_configs()[iso_code]
    language = meta["language"]
    category_names = [c for c, _ in cats]
    print(f"\n{'='*60}\n  Khaya API ({khaya_code}) - {iso_code} ({language})  categories={category_names}\n{'='*60}", flush=True)

    per_category = {}
    cat_wers, cat_cers = [], []
    for category, config in cats:
        samples = load_eval_samples(config, NUM_SAMPLES)
        if not samples:
            continue
        refs = [s["text"] for s in samples]
        print(f"  Category '{category}' ({len(samples)} samples, {MAX_WORKERS} workers)...", flush=True)
        hyps = [""] * len(samples)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            fut = {pool.submit(transcribe, _encode_wav(s["audio"], s["sample_rate"]), khaya_code, key): i
                   for i, s in enumerate(samples)}
            done = 0
            for f in as_completed(fut):
                hyps[fut[f]] = f.result()
                done += 1
                if done % 100 == 0 or done == len(samples):
                    rate = done / (time.time() - t0)
                    print(f"      {done}/{len(samples)}  ({rate:.1f}/s)", flush=True)
        elapsed = time.time() - t0
        wer, cer, valid = _score(refs, hyps)
        save_transcriptions(iso_code, MODEL_ID, category, refs, hyps)
        per_category[category] = {
            "wer": round(wer, 4) if wer is not None else None,
            "cer": round(cer, 4) if cer is not None else None,
            "samples": len(samples), "valid": valid,
            "avg_seconds_per_sample": round(elapsed / max(len(samples), 1), 2),
        }
        if wer is not None:
            cat_wers.append(wer); cat_cers.append(cer)
            print(f"    WER {wer:.2%}  CER {cer:.2%}  (valid {valid}/{len(samples)})", flush=True)

    avg_wer = round(sum(cat_wers) / len(cat_wers), 4) if cat_wers else None
    avg_cer = round(sum(cat_cers) / len(cat_cers), 4) if cat_cers else None
    result = {
        "model": MODEL_ID, "model_url": MODEL_URL, "owner": "KhayaAI",
        "model_class": "non-llm", "params": "API",
        "wer": avg_wer, "cer": avg_cer, "per_category": per_category, "source": "evaluated",
    }
    if avg_wer is None:
        result["error"] = "no_valid_output"
    _save(iso_code, language, category_names, result)
    if avg_wer is not None:
        print(f"  FINAL (avg of {len(cat_wers)} categories): WER {avg_wer:.2%}  CER {avg_cer:.2%}", flush=True)
    return result
