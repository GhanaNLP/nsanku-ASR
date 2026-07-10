"""Orchestrate ASR model evaluation for a single language."""

import csv
import json
import time
from pathlib import Path

import yaml

from .config import RESULTS_FILE, BENCHMARK_DIR, TRANSCRIPTIONS_DIR, NUM_SAMPLES
from .dataset import load_transcripts
from .metrics import compute_metrics
from .models import load_asr_model


def get_language_models(iso_code):
    """Get list of ASR models to evaluate for a language."""
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    lang = data["languages"].get(iso_code)
    if not lang:
        return []
    skip_keywords = ["tts", "text-to-speech", "simba-tts"]
    models = []
    for m in lang.get("asr_models", []):
        if any(kw in m["name"].lower() for kw in skip_keywords):
            continue
        models.append(m)
    return models


def save_benchmark(iso_code, results):
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCHMARK_DIR / f"{iso_code}.yaml"

    existing = {}
    if path.exists():
        with open(path) as f:
            existing = yaml.safe_load(f) or {}

    all_results = existing.get("benchmarks", [])
    seen_models = {r["model"] for r in all_results}

    for r in results:
        if r["model"] not in seen_models:
            all_results.append(r)
            seen_models.add(r["model"])

    out = {
        "iso_639_3": iso_code,
        "num_samples": NUM_SAMPLES,
        "benchmarks": sorted(all_results, key=lambda x: x.get("wer", 1) or 1),
    }

    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  Saved: {path}")


def has_benchmark(iso_code, model_name):
    """Check if a model result already exists for this language."""
    path = BENCHMARK_DIR / f"{iso_code}.yaml"
    if not path.exists():
        return False
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return any(b.get("model") == model_name for b in data.get("benchmarks", []))


def save_transcriptions(iso_code, model_name, references, hypotheses):
    """Save per-sample reference and hypothesis pairs as CSV.

    Columns: sample_id, reference, hypothesis, wer, cer
    Empty/missing hypothesis gets wer=1, cer=1.
    """
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace(":", "_")
    path = TRANSCRIPTIONS_DIR / f"{iso_code}_{safe_name}.csv"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "reference", "hypothesis", "wer", "cer"])
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            if hyp:
                m = compute_metrics(ref, hyp)
                w.writerow([i, ref, hyp, round(m["wer"], 4), round(m["cer"], 4)])
            else:
                w.writerow([i, ref, "", 1.0, 1.0])
    print(f"  Transcriptions: {path}")


def _progress(i, total):
    print(f"    Progress: {i}/{total}")


def evaluate_language(iso_code, model_filter=None, device="cuda:0"):
    """Benchmark all ASR models for a language.

    Each model is loaded, runs 300 samples, then cleaned off GPU before next.

    Args:
        iso_code: ISO 639-3 code
        model_filter: optional string — only benchmark models whose name contains this
        device: torch device
    """
    print(f"\n{'=' * 60}")
    print(f"  Evaluating {iso_code}")
    print(f"{'=' * 60}")

    print(f"  Loading {NUM_SAMPLES} samples...")
    samples = load_transcripts(iso_code)
    if not samples:
        print(f"  No samples loaded for {iso_code}")
        return
    print(f"  Loaded {len(samples)} samples")

    references = [s["text"] for s in samples]
    audio_arrays = [s["audio"] for s in samples]

    models = get_language_models(iso_code)
    if model_filter:
        models = [m for m in models if model_filter.lower() in m["name"].lower()]
    if not models:
        print(f"  No ASR models to evaluate for {iso_code}")
        return

    # Skip models already benchmarked
    bench_path = BENCHMARK_DIR / f"{iso_code}.yaml"
    done_models = set()
    if bench_path.exists():
        with open(bench_path) as f:
            existing = yaml.safe_load(f) or {}
        for b in existing.get("benchmarks", []):
            done_models.add(b["model"])

    pending = [m for m in models if m["name"] not in done_models]
    if not pending:
        print(f"  All models already benchmarked for {iso_code}")
        return

    print(f"  Models: {len(pending)} pending (of {len(models)} total)")
    results = []

    for model_info in pending:
        model_id = model_info["name"]
        params = model_info.get("size", "?")
        print(f"\n  {'-' * 50}")
        print(f"  [{model_id}] ({params})")
        print(f"  {'-' * 50}")

        try:
            model = load_asr_model(model_id, device=device)
        except Exception as load_err:
            err_msg = str(load_err)
            print(f"    Failed to load {model_id}: {err_msg}")
            if err_msg.startswith("GATED:"):
                fail_reason = "gated_repo"
            elif err_msg.startswith("ARCH_UNSUPPORTED:"):
                fail_reason = "architecture_not_supported"
            elif err_msg.startswith("ARCH_UNKNOWN:"):
                fail_reason = "unknown_architecture"
            else:
                fail_reason = "load_failed"
            results.append({
                "model": model_id,
                "model_url": model_info.get("url", f"https://huggingface.co/{model_id}"),
                "params": params,
                "wer": None, "cer": None,
                "error": fail_reason,
                "source": "evaluated",
            })
            continue

        if model is None:
            results.append({
                "model": model_id,
                "model_url": model_info.get("url", f"https://huggingface.co/{model_id}"),
                "params": params,
                "wer": None, "cer": None,
                "error": "load_failed",
                "source": "evaluated",
            })
            continue

        try:
            t0 = time.time()
            hypotheses = model.transcribe_batch(audio_arrays, progress_cb=_progress)
            elapsed = time.time() - t0

            total_wer = total_cer = 0.0
            valid = 0
            per_sample = []
            for ref, hyp in zip(references, hypotheses):
                if hyp:
                    m = compute_metrics(ref, hyp)
                    total_wer += m["wer"]
                    total_cer += m["cer"]
                    per_sample.append(m)
                    valid += 1

            avg_wer = round(total_wer / max(valid, 1), 4)
            avg_cer = round(total_cer / max(valid, 1), 4)
            avg_time = round(elapsed / len(samples), 2)

            result = {
                "model": model_id,
                "model_url": model_info.get("url", f"https://huggingface.co/{model_id}"),
                "params": params,
                "wer": avg_wer,
                "cer": avg_cer,
                "avg_seconds_per_sample": avg_time,
                "source": "evaluated",
            }
            print(f"    WER: {avg_wer:.2%}  CER: {avg_cer:.2%}  ({avg_time}s/sample)")

        except Exception as e:
            print(f"    ERROR during inference: {e}")
            result = {
                "model": model_id,
                "model_url": model_info.get("url", f"https://huggingface.co/{model_id}"),
                "params": params,
                "wer": None, "cer": None,
                "error": str(e),
                "source": "evaluated",
            }
            hypotheses = [""] * len(references)

        results.append(result)
        model.cleanup()

        save_transcriptions(iso_code, model_id, references, hypotheses)

    if results:
        save_benchmark(iso_code, results)

    # Show running leaderboard
    with open(bench_path) as f:
        final = yaml.safe_load(f) or {}
    sorted_bench = sorted(
        [b for b in final.get("benchmarks", []) if b.get("wer") is not None],
        key=lambda x: x["wer"],
    )
    print(f"\n  Leaderboard for {iso_code}:")
    for rank, b in enumerate(sorted_bench[:10], 1):
        print(f"    {rank}. {b['model'][:40]:40s} WER {b['wer']:.2%}  CER {b['cer']:.2%}")

    return results
