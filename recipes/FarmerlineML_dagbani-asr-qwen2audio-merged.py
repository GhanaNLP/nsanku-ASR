"""Evaluation recipe for FarmerlineML/dagbani-asr-qwen2audio-merged.

Architecture: Qwen2-Audio (audio LLM, prompt-driven)
Precision: bf16
Benchmarked languages: dag
Status: not yet benchmarked (queued for the next run)

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated on the next
benchmark run. `build_wrapper(device)` is what the benchmark calls.
"""

from benchmark.models import Qwen2AudioModel

MODEL = "FarmerlineML/dagbani-asr-qwen2audio-merged"

# Qwen2-Audio transcribes by *following an instruction*, so these prompts
# are part of inference — tune them and the next run uses your wording.
# Prompt as documented on the model card by FarmerlineML.
SYSTEM_PROMPT = (
    "You are a Dagbani speech recognition system. "
    "Transcribe the audio exactly as spoken in Dagbani. "
    "Return only the Dagbani transcript, nothing else."
)
USER_PROMPT = "Transcribe this Dagbani audio exactly."
MAX_NEW_TOKENS = 256


def build_wrapper(device="cuda:0", **kwargs):
    return Qwen2AudioModel(
        MODEL, device=device, system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT, max_new_tokens=MAX_NEW_TOKENS, **kwargs,
    )
