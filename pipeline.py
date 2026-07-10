"""Master pipeline: Gemini all 41 langs, then HF models on 8 dedicated-lang models.

Run:  python3 pipeline.py 2>&1 | tee /tmp/nsanku_pipeline.log
"""
import os, sys, time, gc, torch
sys.path.insert(0, ".")

os.environ.setdefault("HF_HOME", "/mnt/volume_d2wey28/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/volume_d2wey28/hf_cache/hub")

from benchmark.gemini import evaluate_gemini
from benchmark.evaluate import evaluate_language

LANGS = [
    "twi","ewe","hau","kbp","dag","dga","fat","nko",
    "any","avn","bud","bim","biv","bib","bwu","ncu",
    "ada","mzw","ffm","acd","gjn","xsm","xon","kma",
    "kus","lef","maw","naw","gur","ntr","nzi","sig",
    "sfw","lip","snw","sil","akp","tpm","kdh","bov","vag",
]
HF_LANGS = ["twi","ewe","hau","kbp","dag","dga","fat","nko"]


def phase1_gemini():
    t0 = time.time()
    for i, iso in enumerate(LANGS, 1):
        print(f"\n===== [{i}/{len(LANGS)}] {iso} =====", flush=True)
        try:
            evaluate_gemini(iso)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
        gc.collect()
    print(f"\nPhase 1 (Gemini) done in {time.time() - t0:.0f}s", flush=True)


def phase2_hf():
    t0 = time.time()
    for iso in HF_LANGS:
        print(f"\n===== HF: {iso} =====", flush=True)
        try:
            evaluate_language(iso)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    print(f"\nPhase 2 (HF) done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("  Phase 1: Gemini — all 41 languages", flush=True)
    print("=" * 60, flush=True)
    phase1_gemini()

    print("=" * 60, flush=True)
    print("  Phase 2: HF models — 8 languages", flush=True)
    print("=" * 60, flush=True)
    phase2_hf()

    print("\nAll done!", flush=True)
