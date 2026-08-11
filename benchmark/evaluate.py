"""Orchestrate ASR model evaluation for a single language.

Scoring: each model is scored on every eval category the language appears in
(bible / jw / finance / unicef). The final WER/CER is the **average across
categories**. Only organization-owned ASR models are benchmarked, loaded with
HuggingFace authentication so gated/org repos are accessible.
"""

import csv
import json
import time
from pathlib import Path

import yaml

from .config import (
    RESULTS_FILE, EVAL_CONFIGS_FILE, BENCHMARK_DIR, TRANSCRIPTIONS_DIR, NUM_SAMPLES,
)
from .dataset import load_eval_samples
from .metrics import compute_metrics
from .models import load_asr_model
from .owners import filter_models
from .recipes import load_recipe


def load_eval_configs():
    with open(EVAL_CONFIGS_FILE) as f:
        return json.load(f)


def language_categories(iso_code):
    """Return list of (category, config) for a language, or [] if not in eval set."""
    cfg = load_eval_configs().get(iso_code)
    if not cfg:
        return []
    return [(c["category"], c["config"]) for c in cfg["categories"]]


# Extra language-tag tokens accepted as an explicit mention of a language.
# Akan (macrolanguage `ak`/`aka`) covers Twi, so Akan-tagged models count for twi.
LANG_TAG_ALIASES = {
    "twi": {"ak", "aka", "akan"},
}


def _iso1_map():
    """iso_639_3 -> iso_639_1, from the language metadata (only where one exists)."""
    import yaml as _yaml
    from .config import LANG_CONFIG
    m = {}
    try:
        meta = _yaml.safe_load(open(LANG_CONFIG))
        for l in meta.get("languages", []):
            if l.get("iso_639_1"):
                m[l["iso_639_3"]] = l["iso_639_1"]
    except Exception:
        pass
    return m


def _all_org_asr_models():
    """Union of every discovered model across languages -> {name: info}."""
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    universe = {}
    for lang in data["languages"].values():
        for m in lang.get("asr_models", []):
            universe.setdefault(m["name"], m)
    return universe


def get_language_models(iso_code):
    """Models to evaluate for a language.

    Eligibility is driven by the models' HF config language tags (not by which
    language page the scrape happened to surface them on): an org-owned ASR model
    is kept only if its config explicitly declares this language (639-3 or 639-1)
    and it is not a generic global base model.
    """
    universe = list(_all_org_asr_models().values())
    iso1 = _iso1_map().get(iso_code)
    codes = {iso_code} | ({iso1} if iso1 else set()) | LANG_TAG_ALIASES.get(iso_code, set())
    return filter_models(universe, iso_codes=codes)


def save_benchmark(iso_code, language, categories, results):
    """Merge results into benchmarks/{iso}.yaml (new per-category schema)."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCHMARK_DIR / f"{iso_code}.yaml"

    existing = {}
    if path.exists():
        with open(path) as f:
            existing = yaml.safe_load(f) or {}

    merged = {r["model"]: r for r in existing.get("benchmarks", [])}
    for r in results:
        merged[r["model"]] = r  # new result overrides

    ranked = sorted(merged.values(), key=lambda x: (x.get("score") is None, x.get("score") or 1e9))
    out = {
        "iso_639_3": iso_code,
        "language": language,
        "num_samples_per_category": NUM_SAMPLES,
        "categories": categories,
        "benchmarks": ranked,
    }
    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  Saved: {path}")


def save_transcriptions(iso_code, model_name, category, references, hypotheses):
    """Save per-sample reference/hypothesis pairs for one (lang, model, category)."""
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    safe = model_name.replace("/", "_").replace(":", "_")
    path = TRANSCRIPTIONS_DIR / f"{iso_code}_{category}_{safe}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "reference", "hypothesis", "wer", "cer"])
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            if hyp:
                m = compute_metrics(ref, hyp)
                w.writerow([i, ref, hyp, round(m["wer"], 4), round(m["cer"], 4)])
            else:
                w.writerow([i, ref, "", 1.0, 1.0])


def _score(references, hypotheses):
    """Return (avg_wer, avg_cer, valid_count) over samples with a hypothesis."""
    total_wer = total_cer = 0.0
    valid = 0
    for ref, hyp in zip(references, hypotheses):
        if hyp:
            m = compute_metrics(ref, hyp)
            total_wer += m["wer"]
            total_cer += m["cer"]
            valid += 1
    if valid == 0:
        return None, None, 0
    return total_wer / valid, total_cer / valid, valid


def _progress(i, total):
    print(f"      Progress: {i}/{total}")


def _done_models(iso_code):
    path = BENCHMARK_DIR / f"{iso_code}.yaml"
    if not path.exists():
        return set()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {b["model"] for b in data.get("benchmarks", []) if b.get("wer") is not None}


def evaluate_language(iso_code, model_filter=None, device="cuda:0"):
    """Benchmark all org ASR models for a language across every eval category.

    Each model is loaded once, scored per category, then final WER/CER is the
    average across the categories the language appears in.
    """
    print(f"\n{'=' * 60}")
    print(f"  Evaluating {iso_code}")
    print(f"{'=' * 60}")

    cats = language_categories(iso_code)
    if not cats:
        print(f"  {iso_code} not present in eval dataset — skipping")
        return
    eval_meta = load_eval_configs()[iso_code]
    language = eval_meta["language"]
    category_names = [c for c, _ in cats]
    print(f"  Language: {language}  Categories: {category_names}")

    # Load samples for every category up-front (reused across all models).
    cat_samples = {}
    for category, config in cats:
        print(f"  Loading {config} ...")
        s = load_eval_samples(config, NUM_SAMPLES)
        if s:
            cat_samples[category] = s
            print(f"    {len(s)} samples")
        else:
            print(f"    (no samples)")
    if not cat_samples:
        print(f"  No samples for any category of {iso_code}")
        return

    models = get_language_models(iso_code)
    if model_filter:
        models = [m for m in models if model_filter.lower() in m["name"].lower()]
    if not models:
        print(f"  No org ASR models to evaluate for {iso_code}")
        return

    done = _done_models(iso_code)
    pending = [m for m in models if m["name"] not in done]
    if not pending:
        print(f"  All {len(models)} org models already benchmarked for {iso_code}")
        return
    print(f"  Models: {len(pending)} pending (of {len(models)} org ASR models)")

    results = []
    for model_info in pending:
        model_id = model_info["name"]
        params = model_info.get("size", "?")
        print(f"\n  {'-' * 50}\n  [{model_id}] ({params})\n  {'-' * 50}")

        try:
            recipe = load_recipe(model_id)
            if recipe:
                print(f"    Recipe overrides: {recipe}")
            model = load_asr_model(model_id, device=device, recipe=recipe)
        except Exception as load_err:
            err = str(load_err)
            reason = ("gated_repo" if err.startswith("GATED:") else
                      "architecture_not_supported" if err.startswith("ARCH_UNSUPPORTED:") else
                      "unknown_architecture" if err.startswith("ARCH_UNKNOWN:") else "load_failed")
            print(f"    Failed to load: {err[:160]}")
            results.append(_error_result(model_info, reason))
            save_benchmark(iso_code, language, category_names, results)
            continue
        if model is None:
            results.append(_error_result(model_info, "load_failed"))
            save_benchmark(iso_code, language, category_names, results)
            continue

        per_category = {}
        cat_wers, cat_cers = [], []
        try:
            for category in category_names:
                samples = cat_samples.get(category)
                if not samples:
                    continue
                refs = [s["text"] for s in samples]
                audio = [s["audio"] for s in samples]
                print(f"    Category '{category}' ({len(samples)} samples)...")
                t0 = time.time()
                hyps = model.transcribe_batch(audio, progress_cb=_progress)
                elapsed = time.time() - t0
                wer, cer, valid = _score(refs, hyps)
                save_transcriptions(iso_code, model_id, category, refs, hyps)
                per_category[category] = {
                    "wer": round(wer, 4) if wer is not None else None,
                    "cer": round(cer, 4) if cer is not None else None,
                    "samples": len(samples),
                    "valid": valid,
                    "avg_seconds_per_sample": round(elapsed / max(len(samples), 1), 2),
                }
                if wer is not None:
                    cat_wers.append(wer)
                    cat_cers.append(cer)
                    print(f"      WER {wer:.2%}  CER {cer:.2%}")
                else:
                    print(f"      no valid output")

            avg_wer = round(sum(cat_wers) / len(cat_wers), 4) if cat_wers else None
            avg_cer = round(sum(cat_cers) / len(cat_cers), 4) if cat_cers else None
            avg_score = round((avg_wer + avg_cer) / 2, 4) if (avg_wer is not None and avg_cer is not None) else None
            result = {
                "model": model_id,
                "model_url": model_info.get("url", f"https://huggingface.co/{model_id}"),
                "owner": model_id.split("/")[0],
                "model_class": "non-llm",
                "params": params,
                "wer": avg_wer,
                "cer": avg_cer,
                "score": avg_score,
                "per_category": per_category,
                "source": "evaluated",
            }
            if avg_wer is not None:
                print(f"    FINAL (avg of {len(cat_wers)} categories): "
                      f"WER {avg_wer:.2%}  CER {avg_cer:.2%}")
            else:
                result["error"] = "no_valid_output"
        except Exception as e:
            print(f"    ERROR during inference: {str(e)[:160]}")
            result = _error_result(model_info, str(e)[:200])
            result["per_category"] = per_category

        results.append(result)
        model.cleanup()
        # Persist after each model so long runs are resumable.
        save_benchmark(iso_code, language, category_names, results)

    _print_leaderboard(iso_code)
    return results


def _error_result(model_info, reason):
    model_id = model_info["name"]
    return {
        "model": model_id,
        "model_url": model_info.get("url", f"https://huggingface.co/{model_id}"),
        "owner": model_id.split("/")[0],
        "model_class": "non-llm",
        "params": model_info.get("size", "?"),
        "wer": None, "cer": None,
        "error": reason,
        "source": "evaluated",
    }


def _print_leaderboard(iso_code):
    path = BENCHMARK_DIR / f"{iso_code}.yaml"
    if not path.exists():
        print(f"\n  No results saved for {iso_code}")
        return
    with open(path) as f:
        final = yaml.safe_load(f) or {}
    ranked = sorted(
        [b for b in final.get("benchmarks", []) if b.get("wer") is not None],
        key=lambda x: x["wer"],
    )
    print(f"\n  Leaderboard for {iso_code} (avg across categories):")
    for rank, b in enumerate(ranked[:10], 1):
        print(f"    {rank}. {b['model'][:42]:42s} WER {b['wer']:.2%}  CER {b['cer']:.2%}")
