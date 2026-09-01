"""Run the sherpa-onnx (ONNX/CPU) track on Modal instead of the local box.

Why: the sherpa track is pure CPU work, and the H200 VM's 20 cores are usually
shared with other jobs — a run there competes for scraps and the wall-clock
becomes unpredictable. Modal rents as many cores as the job wants, and this
track needs no GPU at all, so it is the cheap thing to scale out.

What stays identical to a local run, and must: sample selection comes from the
same benchmark.dataset.load_eval_samples (first NUM_SAMPLES streamed rows of the
config), and the per-language `postprocess` recipe is applied in the container.
Only transcription happens remotely — WER/CER scoring and every file written
under benchmarks/ and transcriptions/ happen locally, through the same
save_benchmark / save_transcriptions the other tracks use. A Modal result and a
local result are therefore comparable.

Fan-out is per (variant, language, category): each task decodes one category in
one container, so a failure costs one category rather than the run.

Setup (one-time):
    pip install modal && modal token new          # already authenticated? skip
    modal run modal_sherpa.py::fetch_models       # populate the model volume

Run:
    modal run modal_sherpa.py::main                       # everything
    modal run modal_sherpa.py::main --langs "dag ewe"     # a few languages
    modal run modal_sherpa.py::main --variant 300m-v2     # one build
"""
import os
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

APP_NAME = "nsanku-sherpa"

# Cores per container. Each decoder process wants 2 threads (measured: the graph
# does not parallelise intra-op), so a container runs CPU_PER_TASK/2 workers.
CPU_PER_TASK = 8
THREADS_PER_WORKER = 2

# Model files live in a Volume so they are downloaded once, not per container.
MODEL_VOLUME = modal.Volume.from_name("nsanku-sherpa-models", create_if_missing=True)
MODEL_DIR = "/models"

# Each build's source repo. The 300M export is published in the GhanaNLP mirror
# (as a tarball); the 1B one only upstream, as a plain directory with ONNX
# external weights.
DOWNLOADS = {
    "300m-v2": {
        "repo": "michsethowusu/sherpa-onnx-omnilingual-asr-1600-languages-ctc-v2",
        "tarball": "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-2026-02-05.tar.bz2",
    },
    "1b-v2": {
        "repo": "csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-2026-02-05",
        "snapshot": "sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-2026-02-05",
    },
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "sherpa-onnx",
        "datasets<4",
        "soundfile",
        "numpy",
        "pyyaml",
        "huggingface_hub",
    )
    # The benchmark package is what guarantees identical sample selection, and
    # recipes/ carries the per-language postprocess hooks.
    .add_local_python_source("benchmark")
    .add_local_dir("recipes", remote_path="/root/recipes")
    .add_local_dir("languages", remote_path="/root/languages")
    .add_local_dir("data", remote_path="/root/data")
)

app = modal.App(APP_NAME, image=image)

# HF_TOKEN for the eval dataset. Created from the local environment:
#   modal secret create nsanku-hf HF_TOKEN=...
hf_secret = modal.Secret.from_name("nsanku-hf")


@app.function(volumes={MODEL_DIR: MODEL_VOLUME}, timeout=3600)
def fetch_models():
    """Populate the model volume. Idempotent — skips builds already present."""
    import subprocess
    from huggingface_hub import hf_hub_download, snapshot_download

    for variant, spec in DOWNLOADS.items():
        target = os.path.join(MODEL_DIR, spec.get("snapshot") or "")
        if "tarball" in spec:
            name = spec["tarball"].replace(".tar.bz2", "")
            target = os.path.join(MODEL_DIR, name)
            if os.path.exists(os.path.join(target, "tokens.txt")):
                print(f"{variant}: already present"); continue
            print(f"{variant}: downloading {spec['tarball']}")
            p = hf_hub_download(spec["repo"], spec["tarball"], local_dir=MODEL_DIR)
            subprocess.run(["tar", "xjf", p, "-C", MODEL_DIR], check=True)
            os.remove(p)
        else:
            if os.path.exists(os.path.join(target, "tokens.txt")):
                print(f"{variant}: already present"); continue
            print(f"{variant}: downloading snapshot")
            snapshot_download(spec["repo"], local_dir=target)
        print(f"{variant}: ready at {target}")
    MODEL_VOLUME.commit()
    return sorted(os.listdir(MODEL_DIR))


@app.function(
    volumes={MODEL_DIR: MODEL_VOLUME},
    cpu=CPU_PER_TASK,
    memory=16384,
    timeout=3600,
    secrets=[hf_secret],
)
def transcribe_category(variant, iso_code, category, config_name):
    """Decode one (variant, language, category) and return the raw hypotheses.

    Returns hypotheses and references rather than a score: scoring stays local
    so that every track goes through the same _score and the same rules about
    empty hypotheses.
    """
    import multiprocessing as mp
    import numpy as np
    import sherpa_onnx

    from benchmark.dataset import load_eval_samples
    from benchmark.recipes import load_lang_recipe, recipe_get
    from benchmark.sherpa import MODELS

    spec = MODELS[variant]
    model_dir = os.path.join(MODEL_DIR, spec["dir"])
    onnx = os.path.join(model_dir, spec["onnx"])
    tokens = os.path.join(model_dir, "tokens.txt")
    if not os.path.exists(onnx):
        raise FileNotFoundError(f"{onnx} missing — run fetch_models first")

    samples = load_eval_samples(config_name)
    if not samples:
        return {"iso": iso_code, "category": category, "variant": variant,
                "refs": [], "hyps": [], "elapsed": 0.0, "audio_sec": 0.0}

    refs = [s["text"] for s in samples]
    items = [(s["audio"], s["sample_rate"]) for s in samples]
    audio_sec = sum(len(s["audio"]) / s["sample_rate"] for s in samples)

    workers = max(1, CPU_PER_TASK // THREADS_PER_WORKER)
    model_id = spec["model"]
    recipe = load_lang_recipe(model_id, iso_code)
    decoding = recipe_get(recipe, "DECODING_METHOD", "greedy_search")
    threads = recipe_get(recipe, "NUM_THREADS", THREADS_PER_WORKER)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_worker,
                  initargs=(onnx, tokens, threads, decoding, model_id, iso_code)) as pool:
        hyps = pool.map(_decode_one, items, chunksize=4)
    elapsed = time.time() - t0

    print(f"{variant} {iso_code}/{category}: {len(hyps)} clips, "
          f"{elapsed:.0f}s, {audio_sec / max(elapsed, 1e-9):.0f}x realtime", flush=True)
    return {"iso": iso_code, "category": category, "variant": variant,
            "refs": refs, "hyps": hyps, "elapsed": elapsed, "audio_sec": audio_sec}


_rec = None
_post = None


def _init_worker(onnx, tokens, num_threads, decoding_method, model_id, iso_code):
    global _rec, _post
    import sherpa_onnx
    from benchmark.recipes import load_lang_recipe, recipe_get
    _rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=onnx, tokens=tokens, num_threads=num_threads,
        decoding_method=decoding_method,
    )
    _post = recipe_get(load_lang_recipe(model_id, iso_code), "postprocess")


def _decode_one(item):
    import numpy as np
    audio, sample_rate = item
    try:
        stream = _rec.create_stream()
        stream.accept_waveform(int(sample_rate), np.asarray(audio, dtype="float32"))
        _rec.decode_stream(stream)
        text = (stream.result.text or "").strip()
    except Exception:
        # An empty hypothesis is excluded from the score, not penalised.
        return ""
    if _post is not None:
        try:
            text = _post(text)
        except Exception:
            pass
    return text


@app.local_entrypoint()
def main(variant: str = "", langs: str = "", force: bool = False):
    """Fan out every (variant, language, category), then score and save locally."""
    from benchmark.evaluate import (language_categories, load_eval_configs,
                                    save_benchmark, save_transcriptions, _score)
    from benchmark.sherpa import MODELS, _build_result, _has_result

    variants = [variant] if variant else sorted(MODELS)
    for v in variants:
        if v not in MODELS:
            raise SystemExit(f"unknown variant {v!r}; choose from {sorted(MODELS)}")
    configs = load_eval_configs()
    isos = langs.split() if langs else list(configs)

    tasks = []
    for v in variants:
        for iso in isos:
            if not force and _has_result(iso, MODELS[v]["model"]):
                print(f"  {v} already done for {iso} - skipping")
                continue
            for category, config_name in language_categories(iso):
                tasks.append((v, iso, category, config_name))

    if not tasks:
        print("nothing to do")
        return
    print(f"dispatching {len(tasks)} category tasks "
          f"({len(variants)} variant(s), {len(isos)} languages)", flush=True)

    t0 = time.time()
    # Collect per (variant, iso) so a language is written once, complete.
    collected = {}
    for res in transcribe_category.starmap(tasks, order_outputs=False):
        key = (res["variant"], res["iso"])
        collected.setdefault(key, {})[res["category"]] = res

    for (v, iso), cats in collected.items():
        spec = MODELS[v]
        meta = configs[iso]
        category_names = [c for c, _ in language_categories(iso)]
        per_category, cat_wers, cat_cers = {}, [], []
        for category in category_names:
            res = cats.get(category)
            if not res or not res["refs"]:
                continue
            wer, cer, valid = _score(res["refs"], res["hyps"])
            save_transcriptions(iso, spec["model"], category, res["refs"], res["hyps"])
            per_category[category] = {
                "wer": round(wer, 4) if wer is not None else None,
                "cer": round(cer, 4) if cer is not None else None,
                "samples": len(res["refs"]),
                "valid": valid,
                "avg_seconds_per_sample": round(
                    res["elapsed"] / max(len(res["refs"]), 1), 2),
            }
            if wer is not None:
                cat_wers.append(wer)
                cat_cers.append(cer)
        result = _build_result(spec, v, per_category, cat_wers, cat_cers)
        save_benchmark(iso, meta["language"], category_names, [result])
        if result["wer"] is not None:
            print(f"  {v} {iso}: WER {result['wer']:.2%}  CER {result['cer']:.2%}")

    print(f"\ndone in {time.time() - t0:.0f}s — {len(collected)} (variant, language) results")
