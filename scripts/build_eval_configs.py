#!/usr/bin/env python3
"""Regenerate data/eval_configs.json from the eval dataset on the Hub.

The file maps each language to the categories it can be scored on:

    {"twi_asante": {"language": "Asante Twi",
                    "categories": [{"category": "bible", "config": "bible_Asante_Twi"}, ...]}}

It used to be maintained by hand, which does not survive adding categories or
splitting a language. Here the dataset is the source of truth: every config is
named `<category>_<label>` and every row carries `iso` and `language`, so both
the grouping and the names are read back from the data rather than guessed from
the config name (which would give `akuapem` for `bible_Akuapem_Twi`).

Usage:
    python scripts/build_eval_configs.py            # rewrite data/eval_configs.json
    python scripts/build_eval_configs.py --dry-run  # print what would change
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.config import EVAL_CONFIGS_FILE, GHANA_SPEECH_EVAL, HF_TOKEN


def _fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem(token=HF_TOKEN or None)


def dataset_configs(dataset):
    """Config names, from the repo's own file tree.

    Deliberately not the datasets-server: its viewer 500s for a while after
    every push (it re-indexes all ~60 configs), which is exactly when this
    script needs to run.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN or None)
    files = api.list_repo_files(dataset, repo_type="dataset")
    return sorted({f.split("/")[0] for f in files if f.endswith(".parquet") and "/" in f})


def config_identity(dataset, config, _split=None):
    """(iso, language) for a config, read straight from its parquet.

    Only those two columns are read, so parquet column pruning keeps this to a
    few KB per config rather than downloading the audio.
    """
    import pyarrow.parquet as pq
    fs = _fs()
    paths = sorted(fs.glob(f"datasets/{dataset}/{config}/*.parquet"))
    if not paths:
        return None, None
    t = pq.read_table(paths[0], columns=["iso", "language"], filesystem=fs)
    d = t.to_pydict()
    if not d["iso"]:
        return None, None
    return (d["iso"][0] or "").strip(), (d["language"][0] or "").strip()


def main(dry_run=False, dataset=GHANA_SPEECH_EVAL):
    langs = defaultdict(lambda: {"language": "", "categories": []})
    skipped = []
    for config in dataset_configs(dataset):
        split = None
        category = config.split("_")[0]
        try:
            iso, language = config_identity(dataset, config, split)
        except Exception as e:
            skipped.append(f"{config} ({type(e).__name__})")
            continue
        if not iso:
            skipped.append(f"{config} (no iso in rows)")
            continue
        entry = langs[iso]
        entry["language"] = entry["language"] or language or iso
        entry["categories"].append({"category": category, "config": config})

    out = {}
    for iso in sorted(langs):
        cats = sorted(langs[iso]["categories"], key=lambda c: c["category"])
        out[iso] = {"language": langs[iso]["language"], "categories": cats}

    old = json.loads(EVAL_CONFIGS_FILE.read_text()) if EVAL_CONFIGS_FILE.exists() else {}
    added = sorted(set(out) - set(old))
    removed = sorted(set(old) - set(out))
    changed = [i for i in sorted(set(out) & set(old))
               if [c["config"] for c in out[i]["categories"]]
               != [c["config"] for c in old[i]["categories"]]]

    print(f"{len(out)} languages from {dataset}")
    if added:
        print("  added:   " + ", ".join(added))
    if removed:
        print("  removed: " + ", ".join(removed))
    for iso in changed:
        before = {c["category"] for c in old[iso]["categories"]}
        after = {c["category"] for c in out[iso]["categories"]}
        print(f"  {iso}: {sorted(before)} -> {sorted(after)}")
    if skipped:
        print("  SKIPPED (not in the output): " + ", ".join(skipped))

    if dry_run:
        print("\n(dry run — nothing written)")
        return out
    EVAL_CONFIGS_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {EVAL_CONFIGS_FILE}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dataset", default=GHANA_SPEECH_EVAL)
    a = ap.parse_args()
    main(dry_run=a.dry_run, dataset=a.dataset)
