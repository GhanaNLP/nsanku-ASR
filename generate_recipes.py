#!/usr/bin/env python3
"""Generate per-model recipe modules (recipes/{owner}_{model}.py).

Each model gets a real Python recipe file that the benchmark imports and runs
directly (see benchmark/recipes.py). A recipe exposes `build_wrapper(device)`
returning the standard wrapper; model authors can edit the file and open a
pull request to change how that model is evaluated on the next run.

Regeneration only creates missing files — an existing recipe is the author's
code and is never overwritten.

Run:  python3 generate_recipes.py
"""
import glob
import sys
from pathlib import Path

import yaml

sys.path.insert(0, ".")
from benchmark.models import _detect_arch, is_ctc_model

ROOT = Path(__file__).parent
RECIPES_DIR = ROOT / "recipes"
REPO_URL = "https://github.com/GhanaNLP/nsanku-ASR"

PRECISION = {
    "ctc": "fp32 (wav2vec2/xls-r/MMS conv encoders crash in bf16 on Hopper)",
    "whisper": "bf16",
    "seq2seq": "bf16",
    "seamless": "-- (not run)",
    "api": "n/a",
}

ARCH_LABEL = {
    "ctc": "CTC (AutoModelForCTC)",
    "whisper": "Whisper seq2seq (AutoModelForSpeechSeq2Seq)",
    "seq2seq": "Seq2Seq (AutoModelForSpeechSeq2Seq)",
    "seamless": "SeamlessM4T - not run (no Ghanaian target-language codes)",
    "api": "Hosted API",
}


def safe_name(model):
    return model.replace("/", "_").replace(":", "_")


def _collect_models():
    models = {}
    for f in glob.glob(str(ROOT / "benchmarks" / "*.yaml")):
        d = yaml.safe_load(open(f)) or {}
        iso = d.get("iso_639_3") or Path(f).stem
        for b in d.get("benchmarks", []):
            m = models.setdefault(
                b["model"],
                {
                    "url": b.get("model_url"),
                    "owner": b.get("owner") or b["model"].split("/")[0],
                    "langs": set(),
                    "passed": False,
                    "best_wer": None,
                    "best_score": None,
                },
            )
            m["langs"].add(iso)
            if b.get("wer") is not None:
                m["passed"] = True
                m["best_wer"] = min(m["best_wer"] or 1e9, b["wer"])
                s = b.get("score") or (b["wer"] + b.get("cer", b["wer"])) / 2
                m["best_score"] = min(m["best_score"] or 1e9, s)
    for m in models.values():
        m["best_wer"] = None if m["best_wer"] is None else round(m["best_wer"], 4)
        m["best_score"] = None if m["best_score"] is None else round(m["best_score"], 4)
    return models


def _khaya_src():
    kh = ROOT / "benchmark" / "khaya.py"
    if not kh.exists():
        return ""
    text = kh.read_text()
    chunks = []
    for start in ("def _encode_wav", "def _transcribe"):
        i = text.find(start)
        if i >= 0:
            j = text.find("\n\n", i)
            chunks.append(text[i : j if j >= 0 else len(text)])
    return "\n".join(chunks)


def _docstring(model_id, model, arch):
    langs = ", ".join(sorted(model["langs"]))
    status = "passed" if model["passed"] else "failed to produce valid output"
    if model["best_wer"] is not None:
        status += " - best avg WER {:.2f}%".format(model["best_wer"] * 100)
        if model["best_score"] is not None:
            status += " (avg WER+CER {:.2f}%)".format(model["best_score"] * 100)
    return "\n".join(
        [
            '"""Evaluation recipe for ' + model_id + ".",
            "",
            "Architecture: " + ARCH_LABEL.get(arch, arch),
            "Precision: " + PRECISION.get(arch, "--"),
            "Benchmarked languages: " + langs,
            "Status: " + status,
            "",
            "Edit this file and open a pull request at",
            REPO_URL + " to change how this model is evaluated on the next",
            "benchmark run. `build_wrapper(device)` is what the benchmark calls.",
            '"""',
        ]
    )


def _body(model_id, arch):
    if arch == "ctc":
        return "\n".join(
            [
                "from benchmark.models import CTCModel",
                "",
                'MODEL = "' + model_id + '"',
                'CTC_DECODER = "greedy"',
                "",
                "",
                "def build_wrapper(device=\"cuda:0\", **kwargs):",
                "    return CTCModel(MODEL, device=device, ctc_decoder=CTC_DECODER, **kwargs)",
                "",
            ]
        )
    if arch in ("whisper", "seq2seq"):
        return "\n".join(
            [
                "from benchmark.models import WhisperModel",
                "",
                'MODEL = "' + model_id + '"',
                "LANGUAGE = None",
                'TASK = "transcribe"',
                "INITIAL_PROMPT = None",
                "",
                "",
                "def build_wrapper(device=\"cuda:0\", **kwargs):",
                "    return WhisperModel(",
                "        MODEL, device=device, language=LANGUAGE,",
                "        task=TASK, initial_prompt=INITIAL_PROMPT, **kwargs,",
                "    )",
                "",
            ]
        )
    if arch == "seamless":
        return "\n".join(
            [
                "# This model uses SeamlessM4T. The benchmark only ships the base",
                "# SeamlessM4T target-language codes, none of which are Ghanaian, so",
                "# reliable transcription was not possible. If this model can",
                "# transcribe Ghanaian languages, add a build_wrapper() below and",
                "# open a pull request.",
                "",
            ]
        )
    # api
    return "\n".join(
        [
            "# This model is a hosted API (benchmark/khaya.py), not run through",
            "# benchmark/models.py. Add a build_wrapper() here only if you want to",
            "# replace the hosted-API evaluation.",
            "",
            "if False:",
            "    from benchmark.khaya import _encode_wav, _transcribe",
            "",
        ]
    )


def render(model_id, model, arch):
    return _docstring(model_id, model, arch) + "\n\n" + _body(model_id, arch)


def main():
    models = _collect_models()
    RECIPES_DIR.mkdir(exist_ok=True)
    created = 0
    skipped = 0
    for model_id, model in sorted(models.items()):
        if model_id == "GhanaNLP/khaya-asr-v3":
            arch = "api"
        else:
            arch = _hf_arch(model_id)
        out = RECIPES_DIR / (safe_name(model_id) + ".py")
        if out.exists():
            skipped += 1
            continue
        out.write_text(render(model_id, model, arch))
        created += 1
        print("  + {}".format(out.name))
    print("\nCreated {} recipes, skipped {} existing (never overwrite author edits).".format(created, skipped))


def _hf_arch(model_id):
    arch = _detect_arch(model_id)
    if arch.startswith("name:"):
        arch = "ctc" if is_ctc_model(model_id) else "whisper"
    return arch


if __name__ == "__main__":
    main()
