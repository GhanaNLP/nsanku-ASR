"""LLM ASR track — Gemini.

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
import base64
from io import BytesIO
from pathlib import Path

import yaml

from .config import NUM_SAMPLES, ROOT
from .dataset import load_eval_samples
from .metrics import compute_metrics
from .evaluate import load_eval_configs, language_categories, save_transcriptions, _score

GEMINI_MODEL = "gemini-3.1-flash-lite"
MODEL_ID = f"google/{GEMINI_MODEL}"
MODEL_URL = f"https://ai.google.dev/gemini-api/docs/models#{GEMINI_MODEL}"
MAX_RETRIES = 3

LLM_BENCHMARK_DIR = ROOT / "benchmarks_llm"


def _encode_wav(audio_array, sample_rate=16000):
    import soundfile as sf
    buf = BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _parse(text):
    m = re.search(r"\[(.*?)\]", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    cleaned = text.strip().strip('"\'')
    return cleaned or None


def _transcribe(wav_bytes, client, language_name=None):
    from google.genai import types
    lang = f"The language is {language_name}. " if language_name else ""
    prompt = (
        "Transcribe the speech in this audio exactly as spoken. " + lang +
        "Put the transcription inside square brackets, e.g. [the man went to the market]. "
        "Output ONLY the bracketed transcription, nothing else."
    )
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[types.Content(parts=[
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                    types.Part.from_text(text=prompt),
                ])],
            )
            out = _parse((resp.text or "").strip())
            if out:
                return out
            last = "empty response"
        except Exception as e:
            last = str(e)
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
    return None  # give up on this sample


def _has_result(iso_code):
    path = LLM_BENCHMARK_DIR / f"{iso_code}.yaml"
    if not path.exists():
        return False
    d = yaml.safe_load(open(path)) or {}
    return any(b.get("wer") is not None for b in d.get("benchmarks", []))


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
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY required.")

    cats = language_categories(iso_code)
    if not cats:
        print(f"  {iso_code} not in eval set — skipping")
        return
    if _has_result(iso_code):
        print(f"  Gemini already done for {iso_code} — skipping")
        return

    meta = load_eval_configs()[iso_code]
    language = meta["language"]
    category_names = [c for c, _ in cats]
    print(f"\n{'=' * 60}\n  Gemini — {iso_code} ({language})  categories={category_names}\n{'=' * 60}")

    client = genai.Client(api_key=api_key)
    per_category = {}
    cat_wers, cat_cers = [], []

    for category, config in cats:
        samples = load_eval_samples(config, NUM_SAMPLES)
        if not samples:
            continue
        refs = [s["text"] for s in samples]
        lang_name = samples[0].get("language") or language
        print(f"  Category '{category}' ({len(samples)} samples)...")
        hyps = []
        t0 = time.time()
        for i, s in enumerate(samples):
            hyps.append(_transcribe(_encode_wav(s["audio"], s["sample_rate"]), client, lang_name) or "")
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"      {i + 1}/{len(samples)}  ({rate:.1f}/s)")
        elapsed = time.time() - t0
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
            print(f"    WER {wer:.2%}  CER {cer:.2%}")

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
        print(f"  FINAL (avg of {len(cat_wers)} categories): WER {avg_wer:.2%}  CER {avg_cer:.2%}")
    return result
