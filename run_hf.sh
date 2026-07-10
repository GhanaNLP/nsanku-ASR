#!/usr/bin/env bash
# Run HF ASR model benchmarks on the H200.
# Picks languages that have Ghana-specific models (>1 dedicated ASR model).
# Usage:
#   ./run_hf.sh                                    # All languages with models
#   ./run_hf.sh twi ewe                                                            # Specific langs
#   LANG_FILTER='whisper' ./run_hf.sh twi                                          # Only whisper models
set -euo pipefail

LANGS=("$@")
if [ ${#LANGS[@]} -eq 0 ]; then
    # Languages with Ghana-specific ASR models (not just espnet/xeus)
    # Manually curated: languages that have >1 model or dedicated fine-tunes
    LANGS=(twi ewe hau kbp dag dga fat nko)
fi

MODEL_FILTER="${MODEL_FILTER:-}"

for iso in "${LANGS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Running HF models for ${iso}"
    echo "============================================================"
    MODEL_FILTER="${MODEL_FILTER}" python3 run_benchmark.py --langs "${iso}" ${MODEL_FILTER:+--models "${MODEL_FILTER}"}
    # Clean up GPU memory between languages
    python3 -c "import torch; torch.cuda.empty_cache()"
done

echo ""
echo "Done!"
echo "Results in benchmarks/{iso}.yaml"
python3 results_summary.py
