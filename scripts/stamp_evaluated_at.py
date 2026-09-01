#!/usr/bin/env python3
"""Stamp each benchmark row with the date its transcriptions were produced.

A score is only true of the system as it was on the day it ran. That matters
most for hosted APIs, whose weights can change without notice and without a
version number — the run date is the only evidence a reader has for what was
actually measured.

Existing rows predate the field, so it is backfilled from git: the latest commit
touching that model's transcription files for that language. Going forward the
runners stamp it directly (see benchmark/evaluate.py and the API tracks).

Usage:
    python scripts/stamp_evaluated_at.py --dry-run
    python scripts/stamp_evaluated_at.py
"""
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.config import BENCHMARK_DIR, TRANSCRIPTIONS_DIR


def git_dates_by_file():
    """{path: latest commit date} for everything under transcriptions/."""
    out = subprocess.run(
        ["git", "log", "--date=short", "--format=__%ad", "--name-only", "--", str(TRANSCRIPTIONS_DIR)],
        capture_output=True, text=True, check=True).stdout
    dates, cur = {}, None
    for line in out.splitlines():
        if line.startswith("__"):
            cur = line[2:]
        elif line.strip() and cur:
            dates.setdefault(line.strip(), cur)     # log is newest-first
    return dates


def main(dry_run=False):
    dates = git_dates_by_file()
    print(f"{len(dates)} transcription file(s) with a commit date")
    changed = missing = 0
    for path in sorted(glob.glob(str(BENCHMARK_DIR / "*.yaml"))):
        if os.path.basename(path).startswith("_"):
            continue
        data = yaml.safe_load(open(path)) or {}
        iso = data.get("iso_639_3")
        touched = False
        for b in data.get("benchmarks", []):
            safe = b["model"].replace("/", "_").replace(":", "_")
            found = [d for f, d in dates.items()
                     if os.path.basename(f).startswith(f"{iso}_") and f.endswith(f"_{safe}.csv")]
            if not found:
                missing += 1
                continue
            latest = max(found)
            if b.get("evaluated_at") != latest:
                b["evaluated_at"] = latest
                touched = True
                changed += 1
        if touched and not dry_run:
            with open(path, "w") as fh:
                yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"{changed} row(s) stamped, {missing} without transcriptions to date from"
          f"{' (dry run — nothing written)' if dry_run else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(dry_run=ap.parse_args().dry_run)
