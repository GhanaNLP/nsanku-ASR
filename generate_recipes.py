#!/usr/bin/env python3
"""Generate per-model recipe files (recipes/{owner}_{model}.md).

Each recipe captures the EXACT inference code used to run a model in this
benchmark (embedded from benchmark/models.py via inspect, so it stays in sync),
plus metadata and the architecture. Model authors can edit a recipe and open a
pull request to correct how their model is evaluated.

Run:  python3 generate_recipes.py
"""
import glob
import inspect
import sys
from pathlib import Path

import yaml

sys.path.insert(0, ".")
from benchmark.models import WhisperModel, CTCModel, _detect_arch, is_ctc_model

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
                },
            )
            m["langs"].add(iso)
            if b.get("wer") is not None:
                m["passed"] = True
                m["best_wer"] = min(m["best_wer"] or 1e9, b["wer"])
    for m in models.values():
        m["best_wer"] = None if m["best_wer"] is None else round(m["best_wer"], 4)
    return models


def _hf_arch(model_id):
    arch = _detect_arch(model_id)
    if arch.startswith("name:"):
        arch = "ctc" if is_ctc_model(model_id) else "whisper"
    return arch


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


def render(model_id, model, arch, note, src):
    langs = ", ".join(sorted(model["langs"]))
    status = "passed" if model["passed"] else "failed to produce valid output"
    wer = ""
    if model["best_wer"] is not None:
        wer = " - best avg WER {:.2f}%".format(model["best_wer"] * 100)
    url = model["url"] or "https://huggingface.co/" + model_id
    precision = PRECISION.get(arch, "--")
    arch_label = ARCH_LABEL.get(arch, arch)

    src_module = "benchmark/khaya.py" if arch == "api" else "benchmark/models.py"
    parts = [
        "# " + model_id,
        "",
        "| Field | Value |",
        "|---|---|",
        "| **Model** | `" + model_id + "` |",
        "| **Owner** | " + model["owner"] + " |",
        "| **URL** | " + url + " |",
        "| **Architecture** | " + arch_label + " |",
        "| **Precision** | " + precision + " |",
        "| **Benchmarked languages** | " + langs + " |",
        "| **Status** | " + status + wer + " |",
        "",
        "## Inference code used",
        "",
        "This model was run with the inference code below from "
        "[`" + src_module + "`](../" + src_module + ").",
        "",
        "```python",
        src,
        "```",
        "",
    ]
    if note:
        parts += ["## Notes", "", note, ""]
    parts += [
        "## Update this recipe",
        "",
        "If this model needs custom inference (custom tokenizer / processor, "
        "language decoding, LM post-processing, etc.) and you believe the WER "
        "above is not representative, edit this file and open a pull request at "
        "[github.com/GhanaNLP/nsanku-ASR](" + REPO_URL + "). The benchmark will "
        "use your updated recipe on the next run.",
        "",
    ]
    return "\n".join(parts)


def main():
    models = _collect_models()
    RECIPES_DIR.mkdir(exist_ok=True)
    written = 0
    for model_id, model in sorted(models.items()):
        note = None
        src = None
        if model_id == "GhanaNLP/khaya-asr-v3":
            arch = "api"
            src = _khaya_src()
        else:
            arch = _hf_arch(model_id)
            if arch == "seamless":
                note = (
                    "This model uses the SeamlessM4T architecture. The benchmark only ships "
                    "the base SeamlessM4T target-language codes, none of which are Ghanaian, "
                    "so reliable transcription was not possible and the model was marked as "
                    "failed.\n\nIf this model can transcribe Ghanaian languages, add the "
                    "required code to this recipe and open a pull request."
                )
            else:
                cls = CTCModel if arch == "ctc" else WhisperModel
                src = inspect.getsource(cls)
        out = RECIPES_DIR / (safe_name(model_id) + ".md")
        out.write_text(render(model_id, model, arch, note, src))
        written += 1
        print("  {}".format(out.name))
    print("\nWrote {} recipes to {}/".format(written, RECIPES_DIR))


if __name__ == "__main__":
    main()
