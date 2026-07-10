"""Load Ghana Speech dataset samples for benchmarking.

Avoids torchcodec dependency by passing decode=False to the Audio feature
and decoding raw bytes with soundfile.
"""

import io

import numpy as np
import soundfile as sf
from datasets import load_dataset, Features, Audio, Value

from .config import GHANA_SPEECH, NUM_SAMPLES, SAMPLE_RATE


def _get_features():
    return Features({
        "id": Value("string"),
        "language": Value("string"),
        "text": Value("string"),
        "duration": Value("float64"),
        "source_file": Value("string"),
        "audio": Audio(sampling_rate=SAMPLE_RATE, decode=False),
    })


def load_transcripts(iso_code):
    """Load up to NUM_SAMPLES samples for a language.

    Returns list of dicts: [{id, text, audio_array, duration}, ...]
    """
    from .config import LANG_TO_CONFIG
    config = LANG_TO_CONFIG.get(iso_code)
    if config is None:
        raise ValueError(f"No dataset config for ISO code: {iso_code}")

    ds = load_dataset(
        GHANA_SPEECH,
        config,
        split="train",
        streaming=True,
        features=_get_features(),
    )

    samples = []
    for i, row in enumerate(ds):
        if i >= NUM_SAMPLES:
            break

        raw = row["audio"]["bytes"]
        if raw is None:
            continue

        audio_array, sr = sf.read(io.BytesIO(raw))
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        samples.append({
            "id": row["id"],
            "text": row["text"],
            "audio": audio_array.astype(np.float32),
            "sample_rate": sr,
            "duration": row.get("duration", len(audio_array) / sr),
            "language": row.get("language"),
        })

    return samples


def load_text_only(iso_code):
    """Load only texts and IDs (no audio) — used for Gemini pre-processing."""
    from .config import LANG_TO_CONFIG
    config = LANG_TO_CONFIG.get(iso_code)
    if config is None:
        raise ValueError(f"No dataset config for ISO code: {iso_code}")

    ds = load_dataset(
        GHANA_SPEECH,
        config,
        split="train",
        streaming=True,
        features=_get_features(),
    )

    samples = []
    for i, row in enumerate(ds):
        if i >= NUM_SAMPLES:
            break
        samples.append({
            "id": row["id"],
            "text": row["text"],
            "duration": row.get("duration", 0.0),
        })

    return samples


def get_audio_only(iso_code):
    """Load audio arrays only — returns (samples, sample_rate)."""
    return load_transcripts(iso_code)
