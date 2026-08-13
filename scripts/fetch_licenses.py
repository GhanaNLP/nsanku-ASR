#!/usr/bin/env python3
"""Fill data/model_licenses.json for every model on the leaderboard.

The licence is read from each repo's cardData on the Hub and cached, so the
leaderboard can show it without a lookup per page load. Models that are hosted
APIs rather than HF repos (the Khaya, Google and Gemini tracks) have no repo to
read and are recorded as "api" — the dashboard renders that as N/A.

Usage:
    python scripts/fetch_licenses.py            # fill what is missing
    python scripts/fetch_licenses.py --refresh  # re-read every model
"""

import argparse
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.config import BENCHMARK_DIR, MODEL_LICENSES_FILE
from benchmark.owners import model_license

# Tracks that are hosted endpoints, not HuggingFace repos.
API_TRACKS = ("KhayaAI/khaya-asr", "Google/speech-recognition", "google/gemini")


def leaderboard_models():
    models = set()
    for f in sorted(glob.glob(str(BENCHMARK_DIR / "*.yaml"))):
        data = yaml.safe_load(open(f)) or {}
        for b in data.get("benchmarks", []):
            models.add(b["model"])
    return sorted(models)


def main(refresh=False):
    cache = json.loads(MODEL_LICENSES_FILE.read_text()) if MODEL_LICENSES_FILE.exists() else {}
    models = leaderboard_models()

    todo = [m for m in models if refresh or not cache.get(m)]
    api = [m for m in todo if m.startswith(API_TRACKS)]
    hub = [m for m in todo if m not in api]
    for m in api:
        cache[m] = "api"

    print(f"{len(models)} models on the leaderboard; {len(hub)} to fetch, {len(api)} API tracks")
    if hub:
        with ThreadPoolExecutor(max_workers=12) as ex:
            for m, lic in zip(hub, ex.map(model_license, hub)):
                cache[m] = lic or ""
                print(f"  {m:60s} {lic or '(none declared)'}")

    MODEL_LICENSES_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    have = sum(1 for m in models if cache.get(m))
    print(f"\nwrote {MODEL_LICENSES_FILE} — {have}/{len(models)} models have a value")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    main(refresh=ap.parse_args().refresh)
