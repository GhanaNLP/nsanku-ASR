#!/usr/bin/env python3
"""Report NEW candidate models for the eval list, for approval before benchmarking.

A candidate is an org-owned, single-language, non-generic ASR model that targets
one of the eval languages and is not already in benchmarks/{iso}.yaml. Candidates
are then bucketed by the model-card gate (see benchmark/owners.py). A declared
license is reported but NOT required.

  APPROVE  — ships a real model card
  BLOCKED  — only a placeholder / missing card (reason shown)

Namespaces in ORG_OVERRIDES (FarmerlineML, GhanaNLP) are exempt from the gate and
always land in APPROVE.

Usage:
    python scripts/curate_models.py            # all eval languages
    python scripts/curate_models.py twi ewe    # only these
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.config import BENCHMARK_DIR, ORG_OVERRIDES, RESULTS_FILE
from benchmark.evaluate import (
    _all_org_asr_models, language_codes, language_name_tokens, load_eval_configs,
)
from benchmark.owners import (
    card_problem, filter_models, model_license, warm_caches_from_universe,
)


def benchmarked(iso):
    path = BENCHMARK_DIR / f"{iso}.yaml"
    if not path.exists():
        return set()
    data = yaml.safe_load(open(path)) or {}
    return {b["model"] for b in data.get("benchmarks", [])}


def main(only=None):
    universe = list(_all_org_asr_models().values())
    with open(RESULTS_FILE) as f:
        warm_caches_from_universe(json.load(f).get("org_models", []))

    configs = load_eval_configs()
    isos = [i for i in configs if not only or i in only]

    # Candidates per language, gate not yet applied.
    candidates, by_lang = set(), {}
    for iso in isos:
        done = benchmarked(iso)
        new = [m["name"] for m in filter_models(universe,
                                                iso_codes=language_codes(iso),
                                                name_tokens=language_name_tokens(iso),
                                                require_card=False)
               if m["name"] not in done]
        by_lang[iso] = new
        candidates.update(new)

    # Warm the license + card caches concurrently.
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(model_license, sorted(candidates)))
        list(ex.map(card_problem, sorted(candidates)))

    def verdict(name):
        if name.split("/")[0] in ORG_OVERRIDES:
            return "", "exempt (ORG_OVERRIDES)"
        lic = model_license(name).strip()
        return card_problem(name), lic or "no license"

    approve, blocked = {}, {}
    for iso, names in by_lang.items():
        for n in names:
            reason, lic = verdict(n)
            (blocked if reason else approve).setdefault(iso, []).append((n, lic, reason))

    lang_name = {i: c["language"] for i, c in configs.items()}

    print(f"=== APPROVE — new models passing all criteria ===")
    total = 0
    for iso in sorted(approve):
        print(f"\n{lang_name.get(iso, iso)} ({iso}):")
        for n, lic, _ in sorted(approve[iso]):
            print(f"  {n:70s} license: {lic}")
            total += 1
    if not approve:
        print("  (none)")
    print(f"\n{total} model(s) ready to benchmark.\n")

    print(f"=== BLOCKED — candidates failing the license/card gate ===")
    nblocked = 0
    for iso in sorted(blocked):
        print(f"\n{lang_name.get(iso, iso)} ({iso}):")
        for n, _, reason in sorted(blocked[iso]):
            print(f"  {n:70s} {reason}")
            nblocked += 1
    if not blocked:
        print("  (none)")
    print(f"\n{nblocked} candidate(s) blocked.")


if __name__ == "__main__":
    main(only=set(sys.argv[1:]) or None)
