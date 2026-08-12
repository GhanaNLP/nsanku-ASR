#!/usr/bin/env python3
"""Drop personal-account models from the leaderboard.

With ORG_ONLY enabled, benchmarks/*.yaml should only list models published by
organizations (plus the ORG_OVERRIDES namespaces such as GhanaNLP, which are
API entries rather than HuggingFace repos). Older result files still contain
entries from personal accounts; this script removes them and re-ranks.

Usage:
    python scripts/apply_org_only.py            # show what would be removed
    python scripts/apply_org_only.py --apply    # rewrite benchmarks/*.yaml
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.config import BENCHMARK_DIR
from benchmark.owners import owner_type


def main(apply=False):
    removed_by_owner = {}
    kept_total = 0

    for path in sorted(BENCHMARK_DIR.glob("*.yaml")):
        data = yaml.safe_load(open(path)) or {}
        entries = data.get("benchmarks", [])
        kept, dropped = [], []
        for e in entries:
            owner = e["model"].split("/")[0]
            (kept if owner_type(owner) == "org" else dropped).append(e)
        kept_total += len(kept)
        for e in dropped:
            removed_by_owner.setdefault(e["model"].split("/")[0], []).append(
                (e["model"], path.stem)
            )
        if not dropped:
            continue
        print(f"{path.name}: {len(entries)} -> {len(kept)} ({len(dropped)} dropped)")
        if apply:
            data["benchmarks"] = sorted(
                kept, key=lambda x: (x.get("score") is None, x.get("score") or 1e9)
            )
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False,
                          allow_unicode=True, sort_keys=False)

    print(f"\nPersonal accounts removed ({sum(len(v) for v in removed_by_owner.values())} entries):")
    for owner in sorted(removed_by_owner):
        langs = ", ".join(sorted({l for _, l in removed_by_owner[owner]}))
        print(f"  {owner:28s} {len(removed_by_owner[owner]):2d} entries  [{langs}]")
    print(f"\nKept: {kept_total} org entries")
    if not apply:
        print("\n(dry run — re-run with --apply to write the files)")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
