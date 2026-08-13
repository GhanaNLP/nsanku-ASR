#!/usr/bin/env python3
"""Generate PER-LANGUAGE recipe modules for the hosted-API / LLM tracks.

The HF model tracks get one recipe per model (generate_recipes.py), because each
model is one model. Khaya, Google ASR and Gemini are the opposite: one endpoint
evaluated on every language, where the thing that actually varies per language is
the API language code (Khaya, Google) or the prompt (Gemini). So each of those
tracks gets one recipe FILE PER LANGUAGE:

    recipes/KhayaAI_khaya-asr-v3__twi_asante.py
    recipes/Google_speech-recognition__twi.py
    recipes/google_gemini-3.6-flash__twi.py

Edit one file to change how that ONE language is evaluated, without touching the
other 15. See benchmark/recipes.py (`load_lang_recipe`) for how they are loaded.

Existing recipes are NEVER overwritten — they are the author's code.

Run:  python3 generate_api_recipes.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark.evaluate import load_eval_configs
from benchmark.gemini import MODEL_ID as GEMINI_ID, GEMINI_MODEL
from benchmark.google import MODEL_ID as GOOGLE_ID, EVAL_TO_GOOGLE
from benchmark.khaya import MODEL_ID as KHAYA_ID, EVAL_TO_KHAYA

RECIPES_DIR = Path(__file__).parent / "recipes"
REPO_URL = "https://github.com/GhanaNLP/nsanku-ASR"


def _header(track, model_id, language, iso, note):
    return f'''"""Evaluation recipe for {model_id} — {language} ({iso}).

Track: {track}
Scope: THIS LANGUAGE ONLY. Every eval language has its own recipe file, so
changing this one does not affect the others.

{note}

Edit this file and open a pull request at
{REPO_URL} to change how {language} is evaluated on the
next benchmark run.
"""'''


def khaya_recipe(iso, language):
    code = EVAL_TO_KHAYA.get(iso)
    note = ("`LANGUAGE_CODE` is the Khaya API's language parameter\n"
            "(POST /asr/v3/transcribe?language=<code>). Set it to None to skip this\n"
            "language. Define `transcribe(wav_bytes, khaya_code, key)` to replace the\n"
            "API call itself.")
    body = [
        _header("Hosted API — Khaya (GhanaNLP)", KHAYA_ID, language, iso, note),
        "",
        "",
        f"LANGUAGE_CODE = {code!r}" if code else
        f"LANGUAGE_CODE = None  # Khaya has no code for {language} yet — set one to enable",
        "",
        "",
        "# def transcribe(wav_bytes, khaya_code, key):",
        "#     from benchmark.khaya import _transcribe",
        "#     return _transcribe(wav_bytes, khaya_code, key)",
        "",
    ]
    return "\n".join(body)


def google_recipe(iso, language):
    code = EVAL_TO_GOOGLE.get(iso)
    note = ("`LANGUAGE_CODE` is the BCP-47 code passed to Google's speech endpoint\n"
            "(`recognize_google(audio, language=...)`). Set it to None to skip this\n"
            "language. Define `transcribe(pcm_bytes, sample_rate, google_code)` to\n"
            "replace the API call itself.")
    body = [
        _header("Hosted API — Google Speech Recognition", GOOGLE_ID, language, iso, note),
        "",
        "",
        f"LANGUAGE_CODE = {code!r}" if code else
        f"LANGUAGE_CODE = None  # Google has no code for {language} — set one to enable",
        "",
        "",
        "# def transcribe(pcm_bytes, sample_rate, google_code):",
        "#     from benchmark.google import _transcribe",
        "#     return _transcribe(pcm_bytes, sample_rate, google_code)",
        "",
    ]
    return "\n".join(body)


def gemini_recipe(iso, language):
    note = ("`PROMPT` is the exact prompt sent with each audio clip. This is the main\n"
            "knob for an LLM track — tune the wording, orthography hints, or examples\n"
            "for this language. The transcription is read back out of the square\n"
            "brackets (see `benchmark.gemini._parse`), so keep that instruction unless\n"
            "you also override `transcribe(wav_bytes, prompt)`.")
    prompt = (
        f'    "Transcribe the speech in this audio exactly as spoken. "\n'
        f'    "The language is {language}. "\n'
        f'    "Put the transcription inside square brackets, e.g. [the man went to the market]. "\n'
        f'    "Output ONLY the bracketed transcription, nothing else."'
    )
    body = [
        _header(f"LLM — Gemini ({GEMINI_MODEL})", GEMINI_ID, language, iso, note),
        "",
        "",
        f"LANGUAGE_NAME = {language!r}",
        "",
        "PROMPT = (",
        prompt,
        ")",
        "",
        "",
        "# def transcribe(wav_bytes, prompt):",
        "#     from benchmark.gemini import _transcribe",
        "#     return _transcribe(wav_bytes, prompt)",
        "",
    ]
    return "\n".join(body)


TRACKS = [
    (KHAYA_ID, khaya_recipe),
    (GOOGLE_ID, google_recipe),
    (GEMINI_ID, gemini_recipe),
]


def main():
    RECIPES_DIR.mkdir(exist_ok=True)
    configs = load_eval_configs()
    created = skipped = 0
    for model_id, render in TRACKS:
        safe = model_id.replace("/", "_").replace(":", "_")
        for iso, cfg in sorted(configs.items()):
            out = RECIPES_DIR / f"{safe}__{iso}.py"
            if out.exists():
                skipped += 1
                continue
            out.write_text(render(iso, cfg["language"]))
            created += 1
            print(f"  + {out.name}")
    print(f"\nCreated {created} per-language recipes, "
          f"skipped {skipped} existing (never overwrite author edits).")


if __name__ == "__main__":
    main()
