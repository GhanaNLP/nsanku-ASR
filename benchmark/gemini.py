"""Gemini ASR evaluation.

Transcribes audio samples using Gemini 3.1 Flash Lite via the Google GenAI API.
Retries on failure, parses transcription from XML tags.
"""

import os
import re
import base64
import time
from io import BytesIO
from pathlib import Path

import numpy as np

# Load .env if present
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    with open(_env) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

    from .config import NUM_SAMPLES, BENCHMARK_DIR
    from .dataset import load_transcripts
    from .metrics import compute_metrics
    from .evaluate import save_benchmark, save_transcriptions, has_benchmark

GEMINI_MODEL = "gemini-3.1-flash-lite"
MAX_RETRIES = 3


def _encode_audio(audio_array, sample_rate=16000):
    import soundfile as sf
    buf = BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_transcription(text):
    """Extract text between [brackets], or return raw text if no brackets."""
    m = re.search(r"\[(.*?)\]", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # No brackets — use raw response as fallback
    cleaned = text.strip().strip('"\'')
    if cleaned:
        return cleaned
    return None


def transcribe_gemini(audio_b64, client, language_name=None):
    """Send audio to Gemini and return transcription text.
    Retries up to MAX_RETRIES times on API errors or missing tags.
    """
    from google.genai import types
    from google.genai import errors as genai_errors

    prompt = (
        'Transcribe the speech in this audio exactly as spoken. '
        f'The language is {language_name}. ' if language_name else ''
        'Put the transcription inside square brackets. '
        'For example: [the man went to the market]. '
        'Output ONLY the bracketed transcription, nothing else.'
    )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_bytes(
                                data=base64.b64decode(audio_b64),
                                mime_type="audio/wav",
                            ),
                            types.Part.from_text(text=prompt),
                        ]
                    )
                ],
            )
            text = response.text.strip()
            transcription = _parse_transcription(text)
            if transcription:
                return transcription
            # Blank response — retry on API issue
            last_error = f"Empty response"
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue

        except (genai_errors.APIError, Exception) as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue

    raise RuntimeError(f"Gemini transcription failed after {MAX_RETRIES} retries: {last_error}")


def evaluate_gemini(iso_code):
    """Evaluate Gemini 3.1 Flash Lite on a language."""
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var.")

    # Skip if already benchmarked
    model_full = f"google/{GEMINI_MODEL}"
    if has_benchmark(iso_code, model_full):
        print(f"  Gemini already done for {iso_code} — skipping")
        return

    client = genai.Client(api_key=api_key)

    print(f"\n{'=' * 60}")
    print(f"  Gemini {GEMINI_MODEL} — {iso_code}")
    print(f"{'=' * 60}")

    print(f"  Loading {NUM_SAMPLES} samples...")
    samples = load_transcripts(iso_code)
    print(f"  Loaded {len(samples)} samples")

    references = [s["text"] for s in samples]
    audio_arrays = [s["audio"] for s in samples]
    first_lang = samples[0].get("language") if samples else None

    hypotheses = []
    errors = 0
    t0 = time.time()

    for i, audio in enumerate(audio_arrays):
        try:
            audio_b64 = _encode_audio(audio)
            text = transcribe_gemini(audio_b64, client, language_name=first_lang)
            if text is None:
                hypotheses.append("")
                errors += 1
            else:
                hypotheses.append(text)
        except Exception as e:
            print(f"    Sample {i}: ERROR — {e}")
            hypotheses.append("")
            errors += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    Progress: {i + 1}/{len(audio_arrays)}  ({rate:.1f} samples/s)")

    elapsed = time.time() - t0

    total_wer = 0.0
    total_cer = 0.0
    valid = 0
    for ref, hyp in zip(references, hypotheses):
        if not hyp:
            continue
        m = compute_metrics(ref, hyp)
        total_wer += m["wer"]
        total_cer += m["cer"]
        valid += 1

    avg_wer = round(total_wer / max(valid, 1), 4)
    avg_cer = round(total_cer / max(valid, 1), 4)

    save_transcriptions(iso_code, f"google/{GEMINI_MODEL}", references, hypotheses)

    result = {
        "model": f"google/{GEMINI_MODEL}",
        "model_url": f"https://ai.google.dev/models/{GEMINI_MODEL}",
        "params": "API",
        "wer": avg_wer,
        "cer": avg_cer,
        "avg_seconds_per_sample": round(elapsed / len(samples), 2),
        "errors": errors,
        "source": "evaluated",
    }

    print(f"\n  Results:")
    print(f"    WER: {avg_wer:.2%}  CER: {avg_cer:.2%}")
    print(f"    Valid: {valid}/{len(samples)}  Errors: {errors}")
    print(f"    Time: {elapsed:.1f}s total ({result['avg_seconds_per_sample']}s/sample)")

    save_benchmark(iso_code, [result])
    return result
