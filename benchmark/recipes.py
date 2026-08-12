"""Recipe loader.

Each model has a per-model recipe module `recipes/{owner}_{model}.py`. The
module is real Python: edit it and open a pull request to change how that model
is evaluated on the next benchmark run. By default it exposes
`build_wrapper(device)` returning the standard wrapper; model authors can
customize anything (subclass, custom `transcribe_batch`, LM post-processing,
tokenizer handling, ...).

The hosted-API / LLM tracks (Khaya, Google, Gemini) are one model evaluated on
many languages, so they additionally get a recipe PER LANGUAGE:
`recipes/{owner}_{model}__{iso}.py`. That is where the per-language knobs live —
the API language code, and for Gemini the prompt — so a contributor can change
how one language is evaluated without touching the others. Use
`load_lang_recipe(model_id, iso)` and `recipe_get(mod, "NAME", default)`.

A broken recipe (import error) does not kill the run: it logs a warning and
falls back to the standard wrapper / built-in defaults.
"""
import importlib.util
from pathlib import Path

RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes"


def recipe_path(model_id: str) -> Path:
    safe = model_id.replace("/", "_").replace(":", "_")
    return RECIPES_DIR / f"{safe}.py"


def lang_recipe_path(model_id: str, iso_code: str) -> Path:
    """Path of the per-language recipe for an API/LLM track."""
    safe = model_id.replace("/", "_").replace(":", "_")
    return RECIPES_DIR / f"{safe}__{iso_code}.py"


def load_recipe(model_id: str):
    """Import and return the model's recipe module, or None if it has none."""
    return _load(recipe_path(model_id))


def load_lang_recipe(model_id: str, iso_code: str):
    """Import the model's recipe for one language, or None if it has none."""
    return _load(lang_recipe_path(model_id, iso_code))


def recipe_get(mod, attr, default=None):
    """Value of `attr` from a recipe module, or `default` when it is not defined.

    An attribute the recipe explicitly sets to None wins over `default`, so a
    recipe can disable a language by setting `LANGUAGE_CODE = None`.
    """
    if mod is None or not hasattr(mod, attr):
        return default
    return getattr(mod, attr)


def _load(path: Path):
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
