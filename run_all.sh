#!/usr/bin/env bash
# nsanku-ASR: Full pipeline — Gemini then HF models — on H200.
# Run in screen:  screen -dmS nsanku ./run_all.sh
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

# Clean slate for Gemini
rm -f benchmarks/twi.yaml

echo ""
echo "============================================================"
echo "  PHASE 1: Gemini 3.1 Flash Lite — All 41 languages"
echo "============================================================"

python3 << 'PYEOF'
from benchmark.gemini import evaluate_gemini
import time
LANGS = ["twi","ewe","hau","kbp","dag","dga","fat","nko","any","avn","bud","bim","biv","bib","bwu","ncu","ada","mzw","ffm","acd","gjn","xsm","xon","kma","kus","lef","maw","naw","gur","ntr","nzi","sig","sfw","lip","snw","sil","akp","tpm","kdh","bov","vag"]
total_start = time.time()
for i, iso in enumerate(LANGS):
    print(f"\n===== [{i+1}/{len(LANGS)}] {iso} =====")
    try: evaluate_gemini(iso)
    except Exception as e: print(f"  FAILED: {e}")
print(f"\nGemini done! {len(LANGS)} languages in {time.time()-total_start:.0f}s")
PYEOF

echo ""
echo "============================================================"
echo "  PHASE 2: HF ASR Models — Languages with dedicated models"
echo "============================================================"

python3 << 'PYEOF'
from benchmark.evaluate import evaluate_language
import time, gc, torch
LANGS = ["twi", "ewe", "hau", "kbp", "dag", "dga", "fat", "nko"]
for iso in LANGS:
    print(f"\n===== HF Models: {iso} =====")
    try:
        evaluate_language(iso)
    except Exception as e:
        print(f"  FAILED: {e}")
    gc.collect()
    torch.cuda.empty_cache()
print("\nHF models done!")
PYEOF

echo ""
echo "============================================================"
echo "  ALL DONE"
echo "============================================================"
python3 results_summary.py
