"""
Search HuggingFace for ASR models tagged with Ghanaian language codes.
Results saved to data/ghana_asr_results.json and data/languages/{iso}.json
"""
import sys
import json
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from fetchers.fetch_huggingface import fetch_models_for_language

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

def main():
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

        # Save per-language
        with open(results_dir / f"{code}.json", "w") as f:
            json.dump(lang_result, f, indent=2, ensure_ascii=False)

        all_results[code] = lang_result
        total_models += total
        if total > 0:
            langs_found += 1

        counts = result.get("counts_by_code", {})
        codes_found = ", ".join(f"{c}:{n}" for c, n in counts.items())
        print(f"    → {total} ASR models (via {codes_found})")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Languages: {len(langs)} unique codes")
    print(f"  Languages with ASR models: {langs_found}/{len(langs)}")
    print(f"  Total ASR models found: {total_models}")
    print(f"  Per-language results: {results_dir}/")

    # Write full results
    summary = {
        "project": "nsanku-ASR",
        "total_languages": len(langs),
        "languages_with_asr_models": langs_found,
        "total_asr_models": total_models,
        "languages": all_results,
    }
    with open(Path(__file__).parent / "data" / "ghana_asr_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Full results: data/ghana_asr_results.json")

if __name__ == "__main__":
    main()
