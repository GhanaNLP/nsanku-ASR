#!/usr/bin/env python3
"""Retry the clips Gemini returned nothing for, and fold any recoveries in.

Gemini leaves gaps: a clip can come back empty from a transient API error, a
safety refusal, or output the bracket parser could not read. `_transcribe`
collapses all three into None, so the cause is invisible after the fact. This
retries only those clips (about 1% of the corpus, versus re-running everything)
and reports what happened to the ones that still fail.

Scoring is unchanged: clips with no transcription stay excluded from WER/CER.
Filling a gap simply moves a clip into the scored set.

Usage:
    python scripts/gemini_fill_gaps.py --dry-run     # what is missing, and where
    python scripts/gemini_fill_gaps.py               # retry and update results
    python scripts/gemini_fill_gaps.py --langs ada bwu
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
csv.field_size_limit(sys.maxsize)

from benchmark.config import BENCHMARK_DIR, NUM_SAMPLES, ROOT, TRANSCRIPTIONS_DIR
from benchmark.dataset import load_eval_samples
from benchmark.evaluate import _score, load_eval_configs, language_categories
from benchmark.gemini import (LLM_BENCHMARK_DIR, MODEL_ID, _encode_wav, _transcribe,
                              default_prompt)
from benchmark.recipes import load_lang_recipe, recipe_get

WORKERS = 8


def transcription_path(iso, category):
    safe = MODEL_ID.replace("/", "_").replace(":", "_")
    return TRANSCRIPTIONS_DIR / f"{iso}_{category}_{safe}.csv"


def read_rows(path):
    with open(path, newline="") as fh:
        r = list(csv.reader(fh))
    return r[0], r[1:]


def gaps_by_language(only=None):
    """{iso: {category: [row indices with no hypothesis]}} from the CSVs."""
    out = defaultdict(dict)
    for f in sorted(glob.glob(str(BENCHMARK_DIR / "*.yaml"))):
        if Path(f).name.startswith("_"):
            continue
        data = yaml.safe_load(open(f)) or {}
        iso = data.get("iso_639_3")
        if not iso or (only and iso not in only):
            continue
        if not any(b.get("model") == MODEL_ID for b in data.get("benchmarks", [])):
            continue
        for category, _cfg in language_categories(iso):
            path = transcription_path(iso, category)
            if not path.exists():
                continue
            _head, rows = read_rows(path)
            missing = [i for i, r in enumerate(rows) if len(r) > 2 and not r[2].strip()]
            if missing:
                out[iso][category] = missing
    return out


def retry_category(iso, category, config, missing, prompt):
    """Re-transcribe the missing indices. Returns (filled, reasons)."""
    samples = load_eval_samples(config, NUM_SAMPLES)
    targets = [i for i in missing if i < len(samples)]
    if not targets:
        return {}, {}

    def one(i):
        wav = _encode_wav(samples[i]["audio"], samples[i]["sample_rate"])
        try:
            return i, _transcribe(wav, prompt), None
        except Exception as e:                      # _transcribe swallows most
            return i, None, f"{type(e).__name__}: {e}"[:120]

    filled, reasons = {}, {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, text, err in pool.map(one, targets):
            if text:
                filled[i] = text
            else:
                reasons[i] = err or "empty after 3 attempts (refusal or unparseable)"
    return filled, reasons


def rescore_and_save(iso, category, filled):
    """Write the recovered text into the CSV and update the Gemini result."""
    from benchmark.metrics import compute_metrics
    path = transcription_path(iso, category)
    head, rows = read_rows(path)
    for i, text in filled.items():
        m = compute_metrics(rows[i][1], text)
        rows[i][2] = text
        rows[i][3] = round(m["wer"], 4)
        rows[i][4] = round(m["cer"], 4)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(rows)

    refs = [r[1] for r in rows]
    hyps = [r[2] for r in rows]
    wer, cer, valid = _score(refs, hyps)

    store = LLM_BENCHMARK_DIR / f"{iso}.yaml"
    data = yaml.safe_load(open(store)) if store.exists() else None
    if not data:
        return wer, cer, valid, False
    for b in data.get("benchmarks", []):
        if b.get("model") != MODEL_ID:
            continue
        pc = b.setdefault("per_category", {})
        entry = pc.setdefault(category, {})
        entry.update({"wer": round(wer, 4), "cer": round(cer, 4),
                      "samples": len(rows), "valid": valid})
        cats = [v for v in pc.values() if v.get("wer") is not None]
        if cats:
            b["wer"] = round(sum(c["wer"] for c in cats) / len(cats), 4)
            b["cer"] = round(sum(c["cer"] for c in cats) / len(cats), 4)
        with open(store, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return wer, cer, valid, True
    return wer, cer, valid, False


def main(dry_run=False, only=None):
    if not dry_run and not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY required (set it or put it in .env)")
    configs = load_eval_configs()
    gaps = gaps_by_language(only)
    total = sum(len(v) for cats in gaps.values() for v in cats.values())
    print(f"{total} untranscribed clip(s) across {len(gaps)} language(s)\n")
    for iso in sorted(gaps):
        for category, missing in sorted(gaps[iso].items()):
            print(f"  {iso:12s} {category:8s} {len(missing):4d} missing")
    if dry_run:
        print("\n(dry run — nothing retried)")
        return

    grand_filled = grand_left = 0
    for iso in sorted(gaps):
        cat_configs = dict(language_categories(iso))
        recipe = load_lang_recipe(MODEL_ID, iso)
        lang = configs[iso]["language"]
        prompt = recipe_get(recipe, "PROMPT", default_prompt(
            recipe_get(recipe, "LANGUAGE_NAME", lang)))
        for category, missing in sorted(gaps[iso].items()):
            print(f"\n{iso}/{category}: retrying {len(missing)} clip(s)...", flush=True)
            filled, reasons = retry_category(iso, category, cat_configs[category],
                                             missing, prompt)
            grand_filled += len(filled)
            grand_left += len(reasons)
            if filled:
                wer, cer, valid, saved = rescore_and_save(iso, category, filled)
                print(f"  recovered {len(filled)}; category now "
                      f"WER {wer:.2%} CER {cer:.2%} on {valid} clips"
                      f"{'' if saved else ' (results file not updated)'}")
            if reasons:
                sample = list(reasons.values())[:2]
                print(f"  still empty: {len(reasons)} — e.g. {sample}")
    print(f"\nrecovered {grand_filled}, still empty {grand_left}")
    if grand_filled:
        print("run merge_llm.py to fold the updated Gemini scores into benchmarks/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--langs", nargs="+")
    a = ap.parse_args()
    main(dry_run=a.dry_run, only=set(a.langs) if a.langs else None)
