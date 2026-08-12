"""LLM ASR track — Gemini (gemini-3.6-flash concurrent with thread-local clients).

Transcribes the ghana-speech-eval audio with Google Gemini (a multimodal LLM) and
scores it per category, averaged across the categories each language appears in —
the same scoring as the non-LLM track. Gemini is a generalist model, so it is run
on ALL eval languages (not just those with dedicated HF ASR models).

Results are written to a SEPARATE store (benchmarks_llm/{iso}.yaml) so this can run
concurrently with the GPU pipeline without racing its YAML writes; a later merge
step folds them into benchmarks/{iso}.yaml tagged model_class="llm".
"""

import os
import re
import time
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from .config import NUM_SAMPLES, ROOT
from .dataset import load_eval_samples
from .evaluate import load_eval_configs, language_categories, save_transcriptions, _score
from .recipes import load_lang_recipe, recipe_get

# Load .env file for GEMINI_API_KEY
env_path = ROOT / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MODEL_ID = f"google/{GEMINI_MODEL}"
MODEL_URL = f"https://ai.google.dev/gemini-api/docs/models#{GEMINI_MODEL}"

MAX_WORKERS = 10
MAX_RETRIES = 3

LLM_BENCHMARK_DIR = ROOT / "benchmarks_llm"
_thread_local = threading.local()


def _get_thread_client():
    if not hasattr(_thread_local, "client"):
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        _thread_local.client = genai.Client(api_key=api_key)
    return _thread_local.client


def _encode_wav(audio_array, sample_rate=16000):
    import soundfile as sf
    buf = BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _parse(text):
    m = re.search(r"\[(.*?)\]", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    cleaned = text.strip().strip("\"'")
    return cleaned or None


def default_prompt(language_name=None):
    """Prompt used when a language has no per-language recipe of its own."""
    lang = f"The language is {language_name}. " if language_name else ""
    return (
        "Transcribe the speech in this audio exactly as spoken. " + lang +
        "Put the transcription inside square brackets, e.g. [the man went to the market]. "
        "Output ONLY the bracketed transcription, nothing else."
    )


def _transcribe(wav_bytes, prompt):
    from google.genai import types
    client = _get_thread_client()
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[types.Content(parts=[
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                    types.Part.from_text(text=prompt),
                ])],
            )
            raw = (resp.text or "").strip()
            out = _parse(raw)
            if out:
                return out
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
    return None


def _transcribe_task(args):
    idx, wav_bytes, prompt, transcribe = args
    res = transcribe(wav_bytes, prompt)
    return idx, res or ""


def _has_result(iso_code):
    """True only if Gemini is scored on every category this language now has.

    The reported WER averages the categories that existed when it ran, so a
    language that has since gained one (lds, waxal) must be re-run rather than
    skipped. Per-category checkpointing means the re-run only pays for the
    categories it is missing.
    """
    path = LLM_BENCHMARK_DIR / f"{iso_code}.yaml"
    if not path.exists():
        return False
    d = yaml.safe_load(open(path)) or {}
    want = {c for c, _ in language_categories(iso_code)}
    for b in d.get("benchmarks", []):
        if b.get("model") != MODEL_ID or b.get("wer") is None:
            continue
        return want <= set(b.get("per_category") or {})
    return False


def _save(iso_code, language, category_names, result):
    LLM_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "iso_639_3": iso_code,
        "language": language,
        "num_samples_per_category": NUM_SAMPLES,
        "categories": category_names,
        "benchmarks": [result],
    }
    with open(LLM_BENCHMARK_DIR / f"{iso_code}.yaml", "w") as f:
        yaml.dump(out, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def evaluate_gemini(iso_code):
    """Evaluate Gemini on one language across all its eval categories."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY required. Make sure it is set or in .env")

    cats = language_categories(iso_code)
    if not cats:
        print(f"  {iso_code} not in eval set - skipping")
        return
    if _has_result(iso_code):
        print(f"  Gemini already done for {iso_code} - skipping")
        return

    meta = load_eval_configs()[iso_code]
    language = meta["language"]
    category_names = [c for c, _ in cats]
    # Per-language recipe (recipes/google_{model}__{iso}.py) owns this language's
    # prompt and may replace the transcribe call entirely.
    recipe = load_lang_recipe(MODEL_ID, iso_code)
    transcribe = recipe_get(recipe, "transcribe", _transcribe)
    print(f"\n{'=' * 60}\n  Gemini ({GEMINI_MODEL}) - {iso_code} ({language})  categories={category_names}\n{'=' * 60}", flush=True)
    if recipe is not None:
        print(f"  recipe: {recipe.__name__.replace('nsanku_recipe_', '')}.py", flush=True)

    # Resume from existing checkpoint if present (per-category granularity)
    existing = {}
    path = LLM_BENCHMARK_DIR / f"{iso_code}.yaml"
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

        samples = load_eval_samples(config, NUM_SAMPLES)
        if not samples:
            continue
        refs = [s["text"] for s in samples]
        lang_name = recipe_get(recipe, "LANGUAGE_NAME",
                               samples[0].get("language") or language)
        prompt = recipe_get(recipe, "PROMPT", default_prompt(lang_name))
        print(f"  Category '{category}' ({len(samples)} samples, {MAX_WORKERS} workers)...", flush=True)

        tasks = []
        for i, s in enumerate(samples):
            wav_bytes = _encode_wav(s["audio"], s["sample_rate"])
            tasks.append((i, wav_bytes, prompt, transcribe))

        hyps = [""] * len(samples)
        done_count = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_transcribe_task, t): t for t in tasks}
            for future in as_completed(futures):
                idx, hyp = future.result()
                hyps[idx] = hyp
                done_count += 1
                if done_count % 100 == 0 or done_count == len(samples):
                    elapsed_so_far = time.time() - t0
                    rate = done_count / elapsed_so_far if elapsed_so_far > 0 else 0
                    eta = (len(samples) - done_count) / rate if rate > 0 else 0
                    print(f"      {done_count}/{len(samples)}  ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)

        elapsed = time.time() - t0
        rate = len(samples) / elapsed if elapsed > 0 else 0
        print(f"      Done {len(samples)} samples in {elapsed:.0f}s ({rate:.1f}/s)", flush=True)

        wer, cer, valid = _score(refs, hyps)
        save_transcriptions(iso_code, MODEL_ID, category, refs, hyps)
        per_category[category] = {
            "wer": round(wer, 4) if wer is not None else None,
            "cer": round(cer, 4) if cer is not None else None,
            "samples": len(samples),
            "valid": valid,
            "avg_seconds_per_sample": round(elapsed / max(len(samples), 1), 2),
        }
        if wer is not None:
            cat_wers.append(wer)
            cat_cers.append(cer)
            print(f"    WER {wer:.2%}  CER {cer:.2%}  (valid {valid}/{len(samples)})", flush=True)

        # Checkpoint after every category so interruptions resume cleanly
        avg_wer = round(sum(cat_wers) / len(cat_wers), 4) if cat_wers else None
        avg_cer = round(sum(cat_cers) / len(cat_cers), 4) if cat_cers else None
        result = {
            "model": MODEL_ID,
            "model_url": MODEL_URL,
            "owner": "google",
            "model_class": "llm",
            "params": "API",
            "wer": avg_wer,
            "cer": avg_cer,
            "per_category": per_category,
            "source": "evaluated",
        }
        if avg_wer is None:
            result["error"] = "no_valid_output"
        _save(iso_code, language, category_names, result)

    avg_wer = round(sum(cat_wers) / len(cat_wers), 4) if cat_wers else None
    avg_cer = round(sum(cat_cers) / len(cat_cers), 4) if cat_cers else None
    result = {
        "model": MODEL_ID,
        "model_url": MODEL_URL,
        "owner": "google",
        "model_class": "llm",
        "params": "API",
        "wer": avg_wer,
        "cer": avg_cer,
        "per_category": per_category,
        "source": "evaluated",
    }
    if avg_wer is None:
        result["error"] = "no_valid_output"
    _save(iso_code, language, category_names, result)
    if avg_wer is not None:
        print(f"  FINAL (avg of {len(cat_wers)} categories): WER {avg_wer:.2%}  CER {avg_cer:.2%}", flush=True)
    return result
