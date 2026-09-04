"""Open ASR track — sherpa-onnx exports of Meta's Omnilingual ASR CTC models.

These are ONNX conversions of the same omniASR CTC checkpoints benchmark/omniasr.py
runs through fairseq2, and they matter for a different reason: they need neither a
GPU nor fairseq2, so a result here says the model is usable on ordinary hardware.
They are downloadable weights, so they score in the open ASR track (non-llm).

A CTC head has no language conditioning — there is no lang argument to pass and
none is used, so the per-language recipes carry a `postprocess` hook instead,
which is where orthography fixes for one language belong.

THREADING — measured, not assumed. On the 20-core H200 box, one recognizer at
num_threads=2 decodes ~7.9x realtime; raising it to 4/10/20 gives 6.2/6.9/4.2x.
The graph does not parallelise intra-op, so throughput comes from running many
small workers instead of one wide one: WORKERS processes of NUM_THREADS each.

Runs in its OWN virtualenv (.venv-sherpa) so the sherpa-onnx/onnxruntime stack
cannot disturb the transformers pins in the main venv — see run_sherpa.py.

Results are merged into benchmarks/{iso}.yaml like any other open-track model.
"""

import multiprocessing as mp
import os
import time

from .config import NUM_SAMPLES, ROOT
from .dataset import load_eval_samples
from .evaluate import (language_categories, load_eval_configs, save_benchmark,
                       save_transcriptions, _score)
from .recipes import load_lang_recipe, recipe_get

# Where the extracted sherpa-onnx model directories live. Override with
# SHERPA_MODEL_DIR when running somewhere other than the H200.
MODEL_ROOT = os.environ.get("SHERPA_MODEL_DIR", "/mnt/volume_d2wey28/models/sherpa")

_V1_300M = "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12"
_V2_300M = "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-2026-02-05"

# Where each build came from. v1 is the original November export; v2 is the
# December retrain, mirrored in the GhanaNLP repo as a tarball.
SOURCE_REPOS = {
    "300m-v1": f"csukuangfj/{_V1_300M}",
    "300m-v2": "michsethowusu/sherpa-onnx-omnilingual-asr-1600-languages-ctc-v2",
}

# Each entry is one leaderboard row. `dir` is relative to MODEL_ROOT.
# Both are the unquantised builds — the id carries no precision claim because
# the export does not state one and sherpa-onnx picks the runtime dtype. Holding
# the size fixed at 300M makes the pair a clean v1-vs-v2 comparison: it isolates
# what the December retrain changed on these languages.
MODELS = {
    "300m-v1": {
        "model": "facebook/sherpa-onnx-omniASR-CTC-300M-v1",
        "dir": _V1_300M,
        "onnx": "model.onnx",
        "params": "0.3B",
    },
    "300m-v2": {
        "model": "facebook/sherpa-onnx-omniASR-CTC-300M-v2",
        "dir": _V2_300M,
        "onnx": "model.onnx",
        "params": "0.3B",
    },
}


def model_url(variant):
    return f"https://huggingface.co/{SOURCE_REPOS[variant]}"

DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) // 2)
DEFAULT_NUM_THREADS = 2

# Worker-process globals: the recognizer is loaded once per worker, not per clip.
_rec = None
_post = None


def model_paths(spec):
    d = os.path.join(MODEL_ROOT, spec["dir"])
    return os.path.join(d, spec["onnx"]), os.path.join(d, "tokens.txt")


def _init_worker(onnx, tokens, num_threads, decoding_method, model_id, iso_code):
    global _rec, _post
    import sherpa_onnx
    _rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=onnx, tokens=tokens, num_threads=num_threads,
        decoding_method=decoding_method,
    )
    # Resolved per worker: the recipe module is not picklable, and each worker
    # imports it once rather than per clip.
    recipe = load_lang_recipe(model_id, iso_code)
    _post = recipe_get(recipe, "postprocess")


def _transcribe_one(item):
    import numpy as np
    audio, sample_rate = item
    try:
        stream = _rec.create_stream()
        stream.accept_waveform(int(sample_rate), np.asarray(audio, dtype="float32"))
        _rec.decode_stream(stream)
        text = (stream.result.text or "").strip()
    except Exception:
        # One unreadable clip must not lose the category; an empty hypothesis is
        # excluded from the score rather than penalised (see _score).
        return ""
    if _post is not None:
        try:
            text = _post(text)
        except Exception:
            pass
    return text


def _has_result(iso_code, model_id):
    """True only if this model is already scored on every category the language has."""
    import yaml
    path = ROOT / "benchmarks" / f"{iso_code}.yaml"
    if not path.exists():
        return False
    d = yaml.safe_load(open(path)) or {}
    want = {c for c, _ in language_categories(iso_code)}
    for b in d.get("benchmarks", []):
        if b.get("model") != model_id or b.get("wer") is None:
            continue
        return want <= set(b.get("per_category") or {})
    return False


def evaluate_sherpa(iso_code, variant, workers=DEFAULT_WORKERS,
                    num_threads=DEFAULT_NUM_THREADS, force=False):
    """Evaluate one sherpa-onnx variant on one language across all its categories."""
    spec = MODELS[variant]
    model_id = spec["model"]
    cats = language_categories(iso_code)
    if not cats:
        print(f"  {iso_code} not in eval set - skipping", flush=True)
        return
    if not force and _has_result(iso_code, model_id):
        print(f"  {variant} already done for {iso_code} - skipping", flush=True)
        return

    onnx, tokens = model_paths(spec)
    if not os.path.exists(onnx):
        raise FileNotFoundError(f"{onnx} not found — extract the tarball under {MODEL_ROOT}")

    meta = load_eval_configs()[iso_code]
    language = meta["language"]
    category_names = [c for c, _ in cats]
    recipe = load_lang_recipe(model_id, iso_code)
    decoding = recipe_get(recipe, "DECODING_METHOD", "greedy_search")
    num_threads = recipe_get(recipe, "NUM_THREADS", num_threads)

    print(f"\n{'=' * 60}\n  sherpa-onnx {variant} - {iso_code} ({language})  "
          f"categories={category_names}  {workers}x{num_threads} threads\n{'=' * 60}", flush=True)

    per_category, cat_wers, cat_cers = {}, [], []
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=workers, initializer=_init_worker,
                    initargs=(onnx, tokens, num_threads, decoding, model_id, iso_code))
    try:
        for category, config in cats:
            samples = load_eval_samples(config, NUM_SAMPLES)
            if not samples:
                continue
            refs = [s["text"] for s in samples]
            items = [(s["audio"], s["sample_rate"]) for s in samples]
            audio_sec = sum(len(s["audio"]) / s["sample_rate"] for s in samples)
            print(f"  Category '{category}' ({len(samples)} samples, "
                  f"{audio_sec / 60:.0f} min audio)...", flush=True)

            t0 = time.time()
            hyps = pool.map(_transcribe_one, items, chunksize=4)
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
                print(f"    WER {wer:.2%}  CER {cer:.2%}  "
                      f"({elapsed:.0f}s, {audio_sec / elapsed:.0f}x realtime)", flush=True)
            else:
                print("    no valid output", flush=True)

            # Checkpoint after every category so an interrupted run resumes.
            save_benchmark(iso_code, language, category_names,
                           [_build_result(spec, variant, per_category, cat_wers, cat_cers)])
    finally:
        pool.close()
        pool.join()

    result = _build_result(spec, variant, per_category, cat_wers, cat_cers)
    save_benchmark(iso_code, language, category_names, [result])
    if result["wer"] is not None:
        print(f"  FINAL (avg of {len(cat_wers)} categories): "
              f"WER {result['wer']:.2%}  CER {result['cer']:.2%}", flush=True)
    return result


def _build_result(spec, variant, per_category, cat_wers, cat_cers):
    avg_wer = round(sum(cat_wers) / len(cat_wers), 4) if cat_wers else None
    avg_cer = round(sum(cat_cers) / len(cat_cers), 4) if cat_cers else None
    # Ranking metric is CER (lower is better). WER/CER kept for display.
    score = round(avg_cer, 4) if avg_cer is not None else None
    result = {
        "model": spec["model"],
        "model_url": model_url(variant),
        "owner": spec["model"].split("/")[0],
        "model_class": "non-llm",
        "params": spec["params"],
        "wer": avg_wer,
        "cer": avg_cer,
        "score": score,
        "per_category": per_category,
        "source": "evaluated",
    }
    if avg_wer is None:
        result["error"] = "no_valid_output"
    return result
