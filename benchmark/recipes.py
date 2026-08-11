"""Recipe front-matter loader.

Each model in this benchmark has a recipe file (`recipes/{owner}_{model}.md`)
that records the exact inference code used to evaluate it. The YAML front-matter
block at the top of a recipe may override evaluation defaults; the benchmark
reads it before running a model, so editing a recipe and opening a pull request
really changes how the model is evaluated on the next run.
"""
from pathlib import Path

import yaml

RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes"

OVERRIDE_KEYS = ("language", "task", "initial_prompt", "ctc_decoder")


def recipe_path(model_id: str) -> Path:
    safe = model_id.replace("/", "_").replace(":", "_")
    return RECIPES_DIR / f"{safe}.md"


def _split_front_matter(text: str):
    if not text.startswith("---"):
        return None, text
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1:])
    return None, text


def load_recipe(model_id: str) -> dict:
    """Return the recipe's front-matter overrides (known keys, nulls dropped)."""
    path = recipe_path(model_id)
    if not path.exists():
        return {}
    fm, _ = _split_front_matter(path.read_text(encoding="utf-8"))
    if fm is None:
        return {}
    try:
        data = yaml.safe_load(fm) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    overrides = {}
    for k in OVERRIDE_KEYS:
        v = data.get(k)
        if v not in (None, "", "null"):
            overrides[k] = v
    return overrides
