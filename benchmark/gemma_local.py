"""LLM ASR track — Gemma 4 12B run LOCALLY on the GPU (bf16, transformers).

Gemma is in the LLM track (a generalist model prompted to transcribe, not an ASR
model), so results land in `benchmarks_llm/{iso}.yaml` with `model_class="llm"`
alongside the Gemini entries, and are scored identically: per category, averaged
across the categories each language appears in.

Why local rather than the Gemini API, which is how the rest of the LLM track
runs: Gemma 4's audio encoder exists only on the E2B, E4B and 12B models, and
the Gemini API exposes only `gemma-4-26b-a4b-it` and `gemma-4-31b-it` — both
audio-less, both returning 400 "Audio input modality is not enabled" on an
audio part. Sending the audio inside an MP4 does not help either: the 26B/31B
process video as image frames and reject the audio track the same way. So the
only audio-capable Gemma has to be run from the weights, which is comfortable on
one H200 (~24GB bf16 against 140GB VRAM).

Thinking on/off is the model's own mechanism, exposed by the Gemma 4 canonical
chat template as `enable_thinking`:
  * True  — injects `<|think|>` at the top of the first system turn.
  * False — pre-fills an empty thought block (`<|channel>thought\n<channel|>`)
             on the model turn, which is what suppresses the "ghost" thought
             channels the 12B/26B/31B otherwise emit even with thinking off.
This is cleaner than the API's `thinking_level`, where `includeThoughts: false`
is silently ignored on Gemma 4 and `thinkingBudget` is rejected outright.

Unlike the API track there is no thread pool: generation is GPU-bound, so clips
are transcribed sequentially and the model is loaded once and reused across
every language of a run.
"""

import gc
import os
import tempfile
import time

import yaml

from .config import NUM_SAMPLES, ROOT
from .dataset import load_eval_samples
from .evaluate import (load_eval_configs, language_categories,
                       save_transcriptions, _score)
from .gemini import LLM_BENCHMARK_DIR, _has_result, _save
from .owners import format_params, model_params
from .recipes import load_lang_recipe, recipe_get

# The audio ASR prompt from the Gemma 4 model card. Kept verbatim: the model was
# tuned against this exact wording, and it asks for a bare transcription, so
# unlike the API track there is no bracket wrapper to parse back out.
ASR_PROMPT = (
    "Transcribe the following speech segment in {lang} into {lang} text. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not "
    "one point seven, and write 3 instead of three."
)

# Transcriptions of <=30s clips; generous enough for a long verse but bounded so
# a degenerate repeat loop cannot stall a 1000-clip category.
MAX_NEW_TOKENS = 256
# Thinking mode needs room for the thought block AND the answer after it. At 256
# the 12B reliably spends the whole budget deliberating ("Wait, let's re-listen")
# and never closes the thought, yielding a response with no content field at all.
MAX_NEW_TOKENS_THINKING = 1024


class GemmaLocalASR:
    """A Gemma 4 unified model held on the GPU, transcribing one clip at a time."""

    def __init__(self, model_id, device="cuda:0", enable_thinking=False):
        import torch
        import transformers

        self.model_id = model_id
        self.device = device
        self.enable_thinking = enable_thinking
        self.processor = transformers.AutoProcessor.from_pretrained(model_id)
        # AutoModelForMultimodalLM is the Gemma 4 unified (text+vision+audio)
        # class; the plain causal-LM class drops the audio tower.
        self.model = transformers.AutoModelForMultimodalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
        ).eval()
        self.max_new_tokens = (MAX_NEW_TOKENS_THINKING if enable_thinking
                               else MAX_NEW_TOKENS)
        # Clips for which the model returned no transcription at all.
        self.no_answer = 0

    def transcribe(self, audio_array, sample_rate, prompt):
        """Transcribe one clip. Returns "" if the model produced nothing.

        An empty return is dropped from the score by `_score` rather than
        counted as a 100% error, matching the rest of the LLM track.
        """
        import torch

        # The processor takes audio by path or URL, so the eval sample's numpy
        # array is staged as a WAV. Written per clip and removed immediately.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            import soundfile as sf
            sf.write(wav_path, audio_array, sample_rate,
                     format="WAV", subtype="PCM_16")
            # Audio goes AFTER the text: the model card is explicit that the
            # prompt must precede the audio part.
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio": wav_path},
                ],
            }]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            ).to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                )
            # skip_special_tokens=False so parse_response can still see the
            # channel markers it needs to split thought from answer.
            decoded = self.processor.decode(
                out[0][input_len:], skip_special_tokens=False)
            # The prompt must be passed as `prefix`: Gemma's template PRE-WRITES
            # part of the model message (with thinking off it emits an already
            # closed empty thought block), and parse_response needs to see that
            # to know where the content field really starts.
            text = _final_answer(self.processor, decoded, inputs["input_ids"])
            if not text:
                self.no_answer += 1
            return text
        finally:
            os.unlink(wav_path)

    def cleanup(self):
        import torch
        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _final_answer(processor, decoded, prefix):
    """Strip the thought channel and special tokens, leaving the transcription.

    With thinking on the model emits a thought channel before its answer, so the
    raw decode is not the transcription. `parse_response` is the processor's own
    splitter, driven by the tokenizer's response_template, and returns
    {'role', 'content'} plus a separate 'thinking' field when a thought block was
    generated — so taking 'content' drops the reasoning without guesswork.
    """
    parsed = None
    try:
        parsed = processor.parse_response(decoded, prefix=prefix)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        # A response that ran out of token budget mid-thought parses to a
        # 'thinking' field with NO 'content'. That is a failed clip: return ""
        # so _score drops it. Falling back to the raw decode here would score
        # the model's reasoning as its transcription — which reads as a ~8x
        # length mismatch and inflates CER past 7.0 rather than looking absent.
        text = parsed.get("content")
        if not text:
            return ""
    elif isinstance(parsed, str):
        text = parsed
    else:
        # Unparseable: recover what follows a closed thought block, then trim at
        # the turn close. Markers match the tokenizer's response_template
        # ('<|channel>thought\n' ... '<channel|>', closed by '<turn|>').
        text = decoded
        if "<|channel>thought" in text and "<channel|>" not in text:
            return ""  # thought never closed -> no answer was produced
        if "<channel|>" in text:
            text = text.rsplit("<channel|>", 1)[-1]
        text = text.split("<turn|>")[0]
    for tok in ("<turn|>", "<eos>", "<end_of_turn>", "<channel|>"):
        text = text.replace(tok, "")
    return " ".join(text.split()).strip()


def evaluate_gemma_local(iso_code, runner, model_id, label=None, force=False):
    """Score one language with an already-loaded Gemma runner.

    runner   — a GemmaLocalASR, loaded once and reused across languages.
    model_id — HF repo id, e.g. 'google/gemma-4-12B-it'.
    label    — suffix distinguishing flavours of one repo ('thinking'/'nothink').
    force    — re-run categories that already have a score.
    """
    record_id = model_id + (f"-{label}" if label else "")
    owner = model_id.split("/")[0] if "/" in model_id else "google"
    model_url = f"https://huggingface.co/{model_id}"

    cats = language_categories(iso_code)
    if not cats:
        print(f"  {iso_code} not in eval set - skipping", flush=True)
        return
    if not force and _has_result(iso_code, record_id):
        print(f"  {record_id} already done for {iso_code} - skipping", flush=True)
        return

    meta = load_eval_configs()[iso_code]
    language = meta["language"]
    category_names = [c for c, _ in cats]
    # Per-language recipe keyed on the base repo, so both thinking flavours of a
    # model share one prompt override (same convention as the API track).
    recipe = load_lang_recipe(model_id, iso_code)
    think_tag = "thinking" if runner.enable_thinking else "nothink"
    print(f"\n{'=' * 60}\n  {record_id} ({think_tag}) - {iso_code} ({language})"
          f"  categories={category_names}\n{'=' * 60}", flush=True)

    params = format_params(model_params(model_id)) or "12B"

    existing = {}
    path = LLM_BENCHMARK_DIR / f"{iso_code}.yaml"
    if path.exists():
        d = yaml.safe_load(open(path)) or {}
        for b in d.get("benchmarks", []):
            if b.get("model") == record_id:
                existing = b.get("per_category") or {}

    per_category = {} if force else dict(existing)
    cat_wers, cat_cers = [], []
    for category, config in cats:
        if (category in per_category
                and per_category[category].get("wer") is not None):
            print(f"  Category '{category}' already done - skipping", flush=True)
            cat_wers.append(per_category[category]["wer"])
            cat_cers.append(per_category[category]["cer"])
            continue

        samples = load_eval_samples(config, NUM_SAMPLES)
        if not samples:
            continue
        refs = [s["text"] for s in samples]
        lang_name = recipe_get(recipe, "LANGUAGE_NAME",
                               samples[0].get("language") or language)
        prompt = recipe_get(recipe, "PROMPT",
                            ASR_PROMPT.format(lang=lang_name))
        print(f"  Category '{category}' ({len(samples)} samples)...", flush=True)

        hyps = []
        runner.no_answer = 0
        t0 = time.time()
        for i, s in enumerate(samples):
            try:
                hyps.append(runner.transcribe(s["audio"], s["sample_rate"], prompt))
            except Exception as e:
                # One bad clip must not lose the category's other 999.
                print(f"      clip {i} failed: {type(e).__name__}: {e}", flush=True)
                hyps.append("")
            n = i + 1
            if n % 50 == 0 or n == len(samples):
                rate = n / max(time.time() - t0, 1e-9)
                eta = (len(samples) - n) / rate if rate > 0 else 0
                print(f"      {n}/{len(samples)}  ({rate:.2f}/s, ETA {eta:.0f}s)",
                      flush=True)
        elapsed = time.time() - t0

        wer, cer, valid = _score(refs, hyps)
        save_transcriptions(iso_code, record_id, category, refs, hyps)
        per_category[category] = {
            "wer": round(wer, 4) if wer is not None else None,
            "cer": round(cer, 4) if cer is not None else None,
            "samples": len(samples),
            "valid": valid,
            "avg_seconds_per_sample": round(elapsed / max(len(samples), 1), 2),
            # Clips excluded from the score because the model returned no
            # transcription. Two observed causes, both real:
            #   * thinking on  — the whole token budget goes on deliberation and
            #     the thought channel never closes, so the parsed response has
            #     no content field.
            #   * thinking off — the model declines outright, generating a single
            #     `<turn|>` token. This is not truncation: retried at a 1024-token
            #     budget, 0 of 5 such clips produced anything.
            # The rate is far from uniform (~3-8% on Twi, ~23% on lower-resource
            # languages like Gikyode and Dangme), so it is recorded per category —
            # a score computed over 77% of the clips is not comparable to one over
            # all of them without saying so.
            **({"no_answer": runner.no_answer} if runner.no_answer else {}),
        }
        if wer is not None:
            cat_wers.append(wer)
            cat_cers.append(cer)
            print(f"    WER {wer:.2%}  CER {cer:.2%}  (valid {valid}/{len(samples)})",
                  flush=True)
        else:
            print(f"    no valid output ({valid}/{len(samples)})", flush=True)
        if runner.no_answer:
            print(f"    {runner.no_answer} clip(s) returned no transcription "
                  f"(excluded from the score)", flush=True)

        # Checkpoint after every category so an interrupted run resumes cleanly.
        _save(iso_code, language, category_names,
              _result(record_id, model_url, owner, params, cat_wers, cat_cers,
                      per_category, think_tag))

    result = _result(record_id, model_url, owner, params, cat_wers, cat_cers,
                     per_category, think_tag)
    _save(iso_code, language, category_names, result)
    if result["wer"] is not None:
        print(f"  FINAL (avg of {len(cat_wers)} categories): "
              f"WER {result['wer']:.2%}  CER {result['cer']:.2%}", flush=True)
    return result


def _result(record_id, model_url, owner, params, cat_wers, cat_cers,
            per_category, think_tag):
    avg_wer = round(sum(cat_wers) / len(cat_wers), 4) if cat_wers else None
    avg_cer = round(sum(cat_cers) / len(cat_cers), 4) if cat_cers else None
    result = {
        "model": record_id,
        "model_url": model_url,
        "owner": owner,
        "model_class": "llm",
        "params": params,
        # Ranking metric is CER (lower is better), same as the non-LLM track.
        # Set here rather than left to the merge step, which only reads `score`
        # and would otherwise rank a scoreless entry last.
        "score": avg_cer,
        # Local weights, not the hosted API — recorded so the two LLM-track
        # sources stay distinguishable in the merged leaderboard.
        "runtime": "local-gpu",
        "thinking": think_tag,
        "wer": avg_wer,
        "cer": avg_cer,
        "per_category": per_category,
        "source": "evaluated",
    }
    if avg_wer is None:
        result["error"] = "no_valid_output"
    return result
