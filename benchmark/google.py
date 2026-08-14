"""Hosted-API ASR track — Google Speech Recognition (free endpoint).

Transcribes the ghana-speech-eval audio with Google's speech recognition
(https://www.google.com — the endpoint used by SpeechRecognition's
`recognize_google`, i.e. Google's free speech-input service). It is scored per
category and averaged, like the other tracks. Results go to
benchmarks_api/{iso}.yaml and are merged into benchmarks/ tagged
model_class="non-llm".

Language codes follow Google's BCP-47: Twi/Akan "ak", Ewe "ee", Ga "gaa".
"""

import os
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
import yaml
import speech_recognition as sr

from .config import NUM_SAMPLES, ROOT
from .dataset import load_eval_samples
from .evaluate import load_eval_configs, language_categories, save_transcriptions, _score
from .recipes import load_lang_recipe, recipe_get

MODEL_ID = "Google/speech-recognition"
MODEL_URL = "https://www.google.com/"
MAX_WORKERS = 4
# An empty result is dropped from the score rather than penalised, so a short
# retry budget silently thins the set a model is judged on instead of hurting
# it. Gemini lost 731 clips this way and every one succeeded on a later retry.
MAX_RETRIES = 6

# eval iso_639_3 -> Google BCP-47 language code
EVAL_TO_GOOGLE = {
    "twi_akuapem": "ak",   # Akan — Google has no dialect-level code
    "twi_asante": "ak",
    "ewe": "ee",   # Ewe
    "gaa": "gaa",  # Ga
}

API_BENCHMARK_DIR = ROOT / "benchmarks_api"


def _encode_pcm(audio_array, sample_rate=16000):
    """float32 [-1,1] array -> int16 PCM bytes for sr.AudioData."""
    return (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _transcribe(pcm_bytes, sample_rate, google_code):
    recognizer = sr.Recognizer()
    audio = sr.AudioData(pcm_bytes, sample_rate, sample_width=2)
    for attempt in range(MAX_RETRIES):
        try:
            text = recognizer.recognize_google(audio, language=google_code)
            return (text or "").strip()
        except sr.UnknownValueError:
            # Google heard speech it could not map to the requested language.
            return ""
        except sr.RequestError as e:
            # Network / quota / rate-limit: back off and retry.
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
            else:
                return ""
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return ""
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


def evaluate_google(iso_code, num_samples=NUM_SAMPLES):
    """Evaluate Google Speech Recognition on one language across its eval categories.

    Checkpoints after every category (writes benchmarks_api/{iso}.yaml), so an
    interrupted run resumes from the last completed category.
    """
    # Per-language recipe (recipes/Google_speech-recognition__{iso}.py) owns this
    # language's BCP-47 code and may replace the transcribe call entirely.
    recipe = load_lang_recipe(MODEL_ID, iso_code)
    google_code = recipe_get(recipe, "LANGUAGE_CODE", EVAL_TO_GOOGLE.get(iso_code))
    transcribe = recipe_get(recipe, "transcribe", _transcribe)
    if not google_code:
        print(f"  {iso_code}: no Google language code configured - skipping")
        return
    cats = language_categories(iso_code)
    if not cats:
        return

    meta = load_eval_configs()[iso_code]
    language = meta["language"]
    category_names = [c for c, _ in cats]
    print(f"\n{'='*60}\n  Google ASR ({google_code}) - {iso_code} ({language})  categories={category_names}\n{'='*60}", flush=True)

    # Resume from existing checkpoint if present
    existing = {}
    path = API_BENCHMARK_DIR / f"{iso_code}.yaml"
    if path.exists():
        d = yaml.safe_load(open(path)) or {}
        for b in d.get("benchmarks", []):
            if b.get("model") == MODEL_ID:
                existing = b.get("per_category") or {}

    per_category = dict(existing)
    cat_wers, cat_cers = [], []
    for category, config in cats:
        if category in per_category and per_category[category].get("wer") is not None:
            print(f"  Category '{category}' already done - skipping", flush=True)
            w = per_category[category]["wer"]
            if w is not None:
                cat_wers.append(w); cat_cers.append(per_category[category]["cer"])
            continue

        samples = load_eval_samples(config, num_samples)
        if not samples:
            continue
        refs = [s["text"] for s in samples]
        print(f"  Category '{category}' ({len(samples)} samples, {MAX_WORKERS} workers)...", flush=True)
        hyps = [""] * len(samples)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            fut = {pool.submit(transcribe, _encode_pcm(s["audio"], s["sample_rate"]), s["sample_rate"], google_code): i
                   for i, s in enumerate(samples)}
            done = 0
            for f in as_completed(fut):
                hyps[fut[f]] = f.result()
                done += 1
                if done % 50 == 0 or done == len(samples):
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

        # Checkpoint after every category so interruptions resume cleanly
        avg_wer = round(sum(cat_wers) / len(cat_wers), 4) if cat_wers else None
        avg_cer = round(sum(cat_cers) / len(cat_cers), 4) if cat_cers else None
        result = {
            "model": MODEL_ID, "model_url": MODEL_URL, "owner": "Google",
            "model_class": "non-llm", "params": "API",
            "wer": avg_wer, "cer": avg_cer, "per_category": per_category, "source": "evaluated",
        }
        if avg_wer is None:
            result["error"] = "no_valid_output"
        _save(iso_code, language, category_names, result)

    avg_wer = round(sum(cat_wers) / len(cat_wers), 4) if cat_wers else None
    avg_cer = round(sum(cat_cers) / len(cat_cers), 4) if cat_cers else None
    if avg_wer is not None:
        print(f"  FINAL (avg of {len(cat_wers)} categories): WER {avg_wer:.2%}  CER {avg_cer:.2%}", flush=True)
    return {"wer": avg_wer, "cer": avg_cer}
