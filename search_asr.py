"""
Search HuggingFace for ASR models tagged with Ghanaian language codes, plus
fetch every ASR model published by the relevant organizations (so org models
are never missed by the per-language page limit).

Results saved to data/ghana_asr_results.json (per-language lists under
`languages`, plus a top-level `org_models` list with all org ASR models).

Usage:
    python3 search_asr.py             # full language search + org scan
    python3 search_asr.py --orgs-only # just refresh org_models (fast)
"""
import sys
import json
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))
from fetchers.fetch_huggingface import fetch_models_for_language, fetch_models_for_org
from benchmark.config import ORG_OVERRIDES, RESULTS_FILE

def load_languages():
    path = Path(__file__).parent / "languages" / "ghana_languages.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data

def deduplicate_languages(data):
    seen = {}
    for lang in data["languages"]:
        code = lang["iso_639_3"]
        if code not in seen:
            seen[code] = {**lang}
        else:
            seen[code]["utterances"] += lang.get("utterances", 0)
            seen[code]["hours"] += lang.get("hours", 0)
    return list(seen.values())

ISO_1_MAP = {"twi": "tw", "ewe": "ee", "hau": "ha", "aka": "ak", "fat": "fat"}


def _existing():
    if RESULTS_FILE.exists():
        try:
            return json.load(open(RESULTS_FILE))
        except Exception:
            return {}
    return {}


def _scan_orgs(known_models):
    """Fetch every ASR model from the known org namespaces + ORG_OVERRIDES.

    `known_models` is an iterable of "owner/model" names already seen; all their
    owners are scanned so org coverage is complete.
    """
    orgs = set(ORG_OVERRIDES)
    orgs.update(m.split("/")[0] for m in known_models if "/" in m)
    org_models = {}
    for org in sorted(orgs):
        try:
            items = fetch_models_for_org(org)
        except Exception as e:
            print(f"      Warning: org scan failed for {org}: {e}")
            continue
        for it in items:
            org_models[it["name"]] = it
    print(f"  Org scan: {len(orgs)} orgs -> {len(org_models)} ASR models")
    return org_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orgs-only", action="store_true",
                    help="skip the per-language scrape; only refresh org_models")
    args = ap.parse_args()

    data = load_languages()
    langs = deduplicate_languages(data)
    langs.sort(key=lambda x: x["hours"], reverse=True)

    results_dir = Path(__file__).parent / "data" / "languages"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  nsanku-ASR — HF ASR Model Search for Ghanaian Languages")
    print("=" * 60)

    all_results = {}
    total_models = 0
    langs_found = 0

    if not args.orgs_only:
        for lang in langs:
            code = lang["iso_639_3"]
            name = lang["name"]
            utts = lang.get("utterances", 0)
            hours = lang.get("hours", 0)
            iso_1 = ISO_1_MAP.get(code)

            print(f"\n  {name:30s} ({code:5s}) — {utts:>7,} utts / {hours:>6.2f}h")

            result = fetch_models_for_language(iso_1, code, "automatic-speech-recognition")
            items = result["items"]
            total = result["total_count"]

            lang_result = {
                "language": name,
                "iso_639_3": code,
                "utterances": utts,
                "hours": hours,
                "total_asr_models": total,
                "models_found": len(items),
                "asr_models": items,
            }

            with open(results_dir / f"{code}.json", "w") as f:
                json.dump(lang_result, f, indent=2, ensure_ascii=False)

            all_results[code] = lang_result
            total_models += total
            if total > 0:
                langs_found += 1

            counts = result.get("counts_by_code", {})
            codes_found = ", ".join(f"{c}:{n}" for c, n in counts.items())
            print(f"    → {total} ASR models (via {codes_found})")
    else:
        existing = _existing()
        all_results = existing.get("languages", {})
        total_models = existing.get("total_asr_models", 0)
        langs_found = existing.get("languages_with_asr_models", 0)
        print(f"  (skipping per-language scrape; using {len(all_results)} cached language results)")

    # Org scan: union of orgs seen in the (new or cached) universe + overrides.
    known = set()
    for lr in all_results.values():
        known.update(m["name"] for m in lr.get("asr_models", []))
    print("\n  Scanning organizations for full model coverage...")
    org_models = _scan_orgs(known)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Languages: {len(langs)} unique codes")
    print(f"  Languages with ASR models: {langs_found}/{len(langs)}")
    print(f"  Total ASR models found (lang search): {total_models}")
    print(f"  Org ASR models found: {len(org_models)}")
    print(f"  Per-language results: {results_dir}/")

    summary = {
        "project": "nsanku-ASR",
        "total_languages": len(langs),
        "languages_with_asr_models": langs_found,
        "total_asr_models": total_models,
        "languages": all_results,
        "org_models": list(org_models.values()),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Full results: {RESULTS_FILE}")

if __name__ == "__main__":
    main()
