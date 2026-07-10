#!/usr/bin/env bash
# Evaluate Gemini 3.1 Flash Lite on Ghanaian languages.
# Usage: ./run_gemini.sh [lang1 lang2 ...]
# Default: all languages with ASR models
#
# Set your API key:
#   export GEMINI_API_KEY="your-key-here"
# Or pass inline:
#   GEMINI_API_KEY="..." ./run_gemini.sh twi ewe dag

set -euo pipefail

# Load API key from .env if not already set
if [ -f .env ]; then
    set -a; source .env; set +a
fi

LANGS=("$@")
if [ ${#LANGS[@]} -eq 0 ]; then
    LANGS=(twi ewe hau kbp dag dga fat nko any avn bud bim biv bib bwu ncu ada mzw ffm acd gjn xsm xon kma kus lef maw naw gur ntr nzi sig sfw lip snw sil akp tpm kdh bov vag)
fi

for iso in "${LANGS[@]}"; do
    echo ""
    python3 -c "
from benchmark.gemini import evaluate_gemini
evaluate_gemini('${iso}')
"
done

echo ""
echo "Done. Results saved to benchmarks/{iso}.yaml"
