"""LLM ASR track — Meta Omnilingual ASR, LLM decoder (omniASR_LLM_7B_v2), GPU.

Unlike Gemini this "LLM" is a LOCAL GPU model: a wav2vec2-style speech encoder
feeding an LLM-inspired transformer decoder with beam search. It is evaluated
on every eval language (it covers 1600+) and scored per category exactly like
the other tracks. Language conditioning uses Meta's own lang ids (e.g.
"dag_Latn") where available; unsupported languages fall back to unconditioned
decoding rather than being dropped.

It runs in its OWN virtualenv (.venv-omniasr): its dependency fairseq2 pins
torch 2.8 while the main benchmark venv runs torch 2.11 — see run_omniasr.py
for the env recipe. The heavy imports are therefore lazy.

Results go to benchmarks_llm/{iso}.yaml MERGED by model id (Gemini's entries
in the same files are preserved) and are folded into benchmarks/{iso}.yaml by
merge_llm.py tagged model_class="llm".

Run:  python3 run_omniasr.py 2>&1 | tee /tmp/nsanku_omniasr.log
      python3 run_omniasr.py --langs dag ewe
"""

import collections
import os
import threading
import time

import yaml

from .config import NUM_SAMPLES, ROOT, SAMPLE_RATE
from .dataset import load_eval_samples
from .evaluate import load_eval_configs, language_categories, save_transcriptions, _score
from .recipes import load_lang_recipe, recipe_get

MODEL_CARD = os.environ.get("OMNIASR_MODEL_CARD", "omniASR_LLM_7B_v2")
MODEL_ID = "facebook/omniASR-LLM-7B-v2"
MODEL_URL = "https://github.com/facebookresearch/omnilingual-asr"

# Clips longer than this are EXCLUDED (not truncated): the pipeline raises on
# anything past 40s, and scoring a truncation against a full-length reference
# would manufacture error. Recorded in the saved failures so coverage loss
# stays visible next to the score.
MAX_AUDIO_SEC = 39.0

DEFAULT_BATCH_SIZE = 16

LLM_BENCHMARK_DIR = ROOT / "benchmarks_llm"

_supported_langs_cache = None
_supported_langs_lock = threading.Lock()


def _supported_lang_ids():
    global _supported_langs_cache
    with _supported_langs_lock:
        if _supported_langs_cache is None:
            from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs
            _supported_langs_cache = set(supported_langs)
    return _supported_langs_cache


def lang_id_for(iso_code, model_id=None):
    """Meta lang id for an eval iso, or None to decode unconditioned.

    The per-language recipe wins when it defines one, so a contributor can
    change the conditioning for a single language of a single checkpoint by
    editing recipes/{model}__{iso}.py. Each checkpoint has its own set of those
    files — the CTC and LLM runs read different recipes even though they share
    this harness.

    Falling back: Twi has no lang id of its own but its macrolanguage Akan does;
    everything else is matched by ISO 639-3 prefix of the "{lang}_{script}" ids.
    """
    if model_id:
        recipe = load_lang_recipe(model_id, iso_code)
        if recipe is not None:
            fn = recipe_get(recipe, "lang_id_for")
            if callable(fn):
                return fn(iso_code)
            if hasattr(recipe, "LANG_ID"):
                return recipe.LANG_ID
    ids = _supported_lang_ids()
    if iso_code.startswith("twi"):
        return "aka_Latn" if "aka_Latn" in ids else None
    for lang in ids:
        if lang.split("_")[0] == iso_code:
            return lang
    return None


class OmniASRLlamaModel:
    """Omnilingual ASR LLM-decoder pipeline (bf16 by default, beam search).

    One pipeline is loaded per process and reused across all languages and
    categories — loading a 7B checkpoint per language would dominate runtime.
    """

    def __init__(self, model_card=None, device="cuda:0", dtype=None,
                 batch_size=DEFAULT_BATCH_SIZE):
        import torch
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        self.model_card = model_card or MODEL_CARD
        self.device = device
        self.batch_size = batch_size
        self.torch = torch
        self.pipeline = ASRInferencePipeline(
            self.model_card,
            device=device,
            dtype=dtype if dtype is not None else torch.bfloat16,
        )

    def transcribe_batch(self, audio_arrays, sample_rate=16000, progress_cb=None,
                         lang_ids=None):
        """Transcribe float32 mono arrays; lang_ids is a scalar id or per-sample list."""
        results = []
        total = len(audio_arrays)
        for start in range(0, total, self.batch_size):
            chunk = []
            for arr in audio_arrays[start:start + self.batch_size]:
                if isinstance(arr, list):
                    arr = self.torch.tensor(arr, dtype=self.torch.float32)
                else:
                    arr = self.torch.as_tensor(arr, dtype=self.torch.float32)
                # 1-D (time,) — the pipeline's collater reads axis 0 as time,
                # so a (1, N) channel-major clip would look like 1 frame.
                chunk.append({"waveform": arr.flatten(), "sample_rate": int(sample_rate)})

            langs = None
            if lang_ids is not None:
                if isinstance(lang_ids, str):
                    langs = [lang_ids] * len(chunk)
                else:
                    langs = [lang_ids[i] if i < len(lang_ids) else None
                             for i in range(start, start + len(chunk))]

            try:
                out = self.pipeline.transcribe(chunk, lang=langs, batch_size=len(chunk))
            except Exception:
                # One bad clip would otherwise lose its whole batch — fall back
                # to clip-by-clip so only genuinely bad clips come back empty.
                out = [""] * len(chunk)
                for j, item in enumerate(chunk):
                    try:
                        res = self.pipeline.transcribe(
                            [item],
                            lang=[langs[j]] if langs else None,
                            batch_size=1,
                        )
                        out[j] = res[0] if res else ""
                    except Exception:
                        pass

            results.extend(t.strip() for t in out)
            if progress_cb:
                progress_cb(min(start + self.batch_size, total), total)
        return results

    def cleanup(self):
        del self.pipeline
        import gc
        gc.collect()
        self.torch.cuda.empty_cache()


# Why clips produced nothing — reset per category, folded into the result.
_failures = collections.Counter()


def reset_failures():
    _failures.clear()


def failure_summary():
    return dict(_failures)


def _progress(i, total):
    if i % 200 == 0 or i == total:
        print(f"      {i}/{total}", flush=True)


def _has_result(iso_code, model_id=MODEL_ID):
    """True only if this model is scored on every category the language now has."""
    path = LLM_BENCHMARK_DIR / f"{iso_code}.yaml"
    if not path.exists():
        return False
    d = yaml.safe_load(open(path)) or {}
    want = {c for c, _ in language_categories(iso_code)}
    for b in d.get("benchmarks", []):
        if b.get("model") != model_id or b.get("wer") is None:
            continue
        return want <= set(b.get("per_category") or {})
    return False


def _load_existing_entry(iso_code, model_id=MODEL_ID):
    path = LLM_BENCHMARK_DIR / f"{iso_code}.yaml"
    if not path.exists():
        return {}
    d = yaml.safe_load(open(path)) or {}
    for b in d.get("benchmarks", []):
        if b.get("model") == model_id:
            return b
    return {}


def _save_merged(iso_code, language, category_names, result, model_id=MODEL_ID):
    """Upsert this model's entry into benchmarks_llm/{iso}.yaml, keeping others."""
    LLM_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = LLM_BENCHMARK_DIR / f"{iso_code}.yaml"
    base = {}
    if path.exists():
        base = yaml.safe_load(open(path)) or {}
    by_model = {b["model"]: b for b in base.get("benchmarks", [])}
    by_model[model_id] = result
    out = {
        "iso_639_3": iso_code,
        "language": language,
        "num_samples_per_category": NUM_SAMPLES,
        "categories": category_names,
        "benchmarks": list(by_model.values()),
    }
    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _build_result(model_id, params, per_category, cat_wers, cat_cers,
                  model_class="llm"):
    """Assemble one benchmark entry.

    model_class decides the leaderboard track: the LLM-decoder checkpoints are
    "llm", but the same pipeline also serves the CTC checkpoints, which are
    plain downloadable ASR models and belong in the open ASR track ("non-llm").
    """
    avg_wer = round(sum(cat_wers) / len(cat_wers), 4) if cat_wers else None
    avg_cer = round(sum(cat_cers) / len(cat_cers), 4) if cat_cers else None
    result = {
        "model": model_id,
        "model_url": MODEL_URL,
        "owner": "facebook",
        "model_class": model_class,
        "params": params,
        "wer": avg_wer,
        "cer": avg_cer,
        "per_category": per_category,
        "source": "evaluated",
    }
    if avg_wer is None:
        result["error"] = "no_valid_output"
    return result


def evaluate_omniasr(iso_code, model=None, force=False, model_id=MODEL_ID,
                     card=MODEL_CARD, params="7B", model_class="llm"):
    """Evaluate an OmniASR checkpoint on one language across all its categories."""
    cats = language_categories(iso_code)
    if not cats:
        print(f"  {iso_code} not in eval set - skipping")
        return
    if not force and _has_result(iso_code, model_id):
        print(f"  OmniASR ({card}) already done for {iso_code} - skipping")
        return

    meta = load_eval_configs()[iso_code]
    language = meta["language"]
    category_names = [c for c, _ in cats]
    cond = lang_id_for(iso_code, model_id)
    print(f"\n{'=' * 60}\n  OmniASR ({card}) - {iso_code} ({language})  "
          f"categories={category_names}  lang_cond={cond or 'none'}\n{'=' * 60}", flush=True)

    # Resume from the existing checkpoint if present (per-category granularity)
    existing = _load_existing_entry(iso_code, model_id)
    per_category = dict(existing.get("per_category") or {})

    if model is None:
        model = get_shared_model(card=card)

    cat_wers, cat_cers = [], []
    # Categories already scored count toward the average on resume.
    for category in category_names:
        got = per_category.get(category)
        if got and got.get("wer") is not None:
            print(f"  Category '{category}' already done - skipping", flush=True)
            cat_wers.append(got["wer"])
            cat_cers.append(got["cer"])

    for category, config in cats:
        if per_category.get(category, {}).get("wer") is not None:
            continue

        samples = load_eval_samples(config, NUM_SAMPLES)
        if not samples:
            continue

        reset_failures()
        kept, dropped_long = [], 0
        for s in samples:
            dur = len(s["audio"]) / s["sample_rate"]
            if dur > MAX_AUDIO_SEC:
                dropped_long += 1
            else:
                kept.append(s)
        if dropped_long:
            _failures[f"audio_too_long (> {MAX_AUDIO_SEC:.0f}s)"] += dropped_long

        refs = [s["text"] for s in kept]
        audio = [s["audio"] for s in kept]
        print(f"  Category '{category}' ({len(kept)} samples"
              f"{f', {dropped_long} too long' if dropped_long else ''})...", flush=True)

        t0 = time.time()
        hyps = model.transcribe_batch(
            audio, SAMPLE_RATE, progress_cb=_progress,
            lang_ids=cond,
        )
        elapsed = time.time() - t0

        wer, cer, valid = _score(refs, hyps)
        save_transcriptions(iso_code, model_id, category, refs, hyps)
        failures = failure_summary()
        per_category[category] = {
            "wer": round(wer, 4) if wer is not None else None,
            "cer": round(cer, 4) if cer is not None else None,
            "samples": len(samples),
            "valid": valid,
            "avg_seconds_per_sample": round(elapsed / max(len(samples), 1), 2),
            **({"failures": failures} if failures else {}),
        }
        if failures:
            detail = ", ".join(f"{k}: {v}" for k, v in sorted(failures.items()))
            print(f"    {len(samples) - valid} clip(s) unscored - {detail}", flush=True)
        if wer is not None:
            cat_wers.append(wer)
            cat_cers.append(cer)
            print(f"    WER {wer:.2%}  CER {cer:.2%}  "
                  f"({elapsed:.0f}s, {len(samples) / elapsed:.2f}/s)", flush=True)

        # Checkpoint after every category so interruptions resume cleanly
        result = _build_result(model_id, params, per_category, cat_wers, cat_cers,
                              model_class)
        _save_merged(iso_code, language, category_names, result, model_id)

    result = _build_result(model_id, params, per_category, cat_wers, cat_cers,
                              model_class)
    _save_merged(iso_code, language, category_names, result, model_id)
    if result["wer"] is not None:
        print(f"  FINAL (avg of {len(cat_wers)} categories): "
              f"WER {result['wer']:.2%}  CER {result['cer']:.2%}", flush=True)
    return result


_shared_model = None


def get_shared_model(card=MODEL_CARD):
    """Load (once) the pipeline reused across all languages in a run.

    Keyed on the card: a run that switches checkpoints must not silently keep
    scoring with the first one that happened to be loaded.
    """
    global _shared_model
    if _shared_model is not None and _shared_model.model_card != card:
        drop_shared_model()
    if _shared_model is None:
        _shared_model = OmniASRLlamaModel(card)
    return _shared_model


def drop_shared_model():
    global _shared_model
    if _shared_model is not None:
        _shared_model.cleanup()
        _shared_model = None
