"""Recipe loader.

Each model has a per-model recipe module `recipes/{owner}_{model}.py`. The
module is real Python: edit it and open a pull request to change how that model
is evaluated on the next benchmark run. By default it exposes
`build_wrapper(device)` returning the standard wrapper; model authors can
customize anything (subclass, custom `transcribe_batch`, LM post-processing,
tokenizer handling, ...).

A broken recipe (import error) does not kill the run: it logs a warning and
falls back to the standard wrapper.
"""
import importlib.util
from pathlib import Path

RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes"


def recipe_path(model_id: str) -> Path:
    safe = model_id.replace("/", "_").replace(":", "_")
    return RECIPES_DIR / f"{safe}.py"


def load_recipe(model_id: str):
    """Import and return the model's recipe module, or None if it has none."""
    path = recipe_path(model_id)
    if not path.exists():
        return None
    name = "nsanku_recipe_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"    WARN: recipe {path.name} failed to import ({e}); using defaults")
        return None
    return mod
