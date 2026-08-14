#!/usr/bin/env python3
"""Recompute a model's scores in benchmarks/ from its transcription CSVs.

The CSVs are the record of what a model actually produced, so they can rebuild
the leaderboard numbers without re-running inference. Useful when the
transcriptions change but the scores have not — after filling in clips that
came back empty, for instance — or when the intermediate results store
(benchmarks_llm/, benchmarks_api/) is missing, since those are gitignored and
do not survive a fresh clone.

Scoring matches benchmark.evaluate._score exactly: clips with no hypothesis are
excluded rather than penalised, and `valid` records how many counted.

Usage:
    python scripts/rescore_from_transcriptions.py --model google/gemini-3.6-flash --dry-run
    python scripts/rescore_from_transcriptions.py --model google/gemini-3.6-flash
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
csv.field_size_limit(sys.maxsize)

from benchmark.config import BENCHMARK_DIR, TRANSCRIPTIONS_DIR
from benchmark.evaluate import language_categories
from benchmark.metrics import compute_metrics


def score_csv(path):
    """(wer, cer, valid, total) over the rows of one transcription file."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))[1:]
    tw = tc = 0.0
    valid = 0
    for r in rows:
        if len(r) < 3 or not r[2].strip():
            continue                      # no hypothesis: excluded, not scored
        m = compute_metrics(r[1], r[2])
        tw += m["wer"]
        tc += m["cer"]
        valid += 1
    if not valid:
        return None, None, 0, len(rows)
    return tw / valid, tc / valid, valid, len(rows)


def main(model, dry_run=False):
    safe = model.replace("/", "_").replace(":", "_")
    changed = 0
    for path in sorted(glob.glob(str(BENCHMARK_DIR / "*.yaml"))):
        if Path(path).name.startswith("_"):
            continue
        data = yaml.safe_load(open(path)) or {}
        iso = data.get("iso_639_3")
        entry = next((b for b in data.get("benchmarks", []) if b.get("model") == model), None)
        if not iso or entry is None:
            continue

        per_cat, updates = {}, []
        for category, _cfg in language_categories(iso):
            f = TRANSCRIPTIONS_DIR / f"{iso}_{category}_{safe}.csv"
            if not f.exists():
                old = (entry.get("per_category") or {}).get(category)
                if old:
                    per_cat[category] = old      # keep what we cannot recompute
                continue
            wer, cer, valid, total = score_csv(f)
            if wer is None:
                continue
            old = (entry.get("per_category") or {}).get(category) or {}
            per_cat[category] = {
                "wer": round(wer, 4), "cer": round(cer, 4),
                "samples": total, "valid": valid,
                **({"avg_seconds_per_sample": old["avg_seconds_per_sample"]}
                   if "avg_seconds_per_sample" in old else {}),
            }
            if old.get("valid") != valid or old.get("wer") != round(wer, 4):
                updates.append(f"{category}: {old.get('wer')}→{round(wer,4)} "
                               f"on {old.get('valid')}→{valid} clips")
        if not per_cat:
            continue
        scored = [v for v in per_cat.values() if v.get("wer") is not None]
        new_wer = round(sum(v["wer"] for v in scored) / len(scored), 4)
        new_cer = round(sum(v["cer"] for v in scored) / len(scored), 4)
        if updates or entry.get("wer") != new_wer:
            print(f"{iso}: avg {entry.get('wer')} → {new_wer}")
            for u in updates:
                print(f"    {u}")
            entry["per_category"] = per_cat
            entry["wer"], entry["cer"] = new_wer, new_cer
            entry["score"] = round((new_wer + new_cer) / 2, 4)
            data["benchmarks"] = sorted(
                data["benchmarks"],
                key=lambda x: (x.get("score") is None, x.get("score") or 1e9))
            changed += 1
            if not dry_run:
                with open(path, "w") as fh:
                    yaml.dump(data, fh, default_flow_style=False,
                              allow_unicode=True, sort_keys=False)
    print(f"\n{changed} language file(s) {'would change' if dry_run else 'updated'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.model, dry_run=a.dry_run)
