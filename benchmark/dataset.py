"""Load ghana-speech-eval samples for benchmarking.

Avoids torchcodec dependency by passing decode=False to the Audio feature
and decoding raw bytes with soundfile.
"""

import io
import os

import numpy as np
import soundfile as sf
from datasets import load_dataset, Features, Audio, Value

from .config import GHANA_SPEECH_EVAL, NUM_SAMPLES, SAMPLE_RATE, HF_TOKEN


def _eval_features():
    """Features for ghananlpcommunity/ghana-speech-eval configs."""
    return Features({
        "audio": Audio(sampling_rate=SAMPLE_RATE, decode=False),
        "text": Value("string"),
        "language": Value("string"),
        "country": Value("string"),
        "length": Value("float64"),
        "iso": Value("string"),
        "subset": Value("string"),
    })


def _load_dataset(config_name):
    """Stream the eval config, preferring a local snapshot when configured.

    Set NSANKU_EVAL_LOCAL_DIR to a snapshot of the dataset repo (e.g.
    /mnt/.../hf_datasets/ghana-speech-eval) to read parquet from disk instead
    of the HF CDN — the VM's CDN route drops connections, which used to stall
    category loading in endless retries. Streaming semantics are unchanged.
    """
    local_dir = os.environ.get("NSANKU_EVAL_LOCAL_DIR", "")
    if local_dir:
        import glob
        files = sorted(glob.glob(os.path.join(local_dir, config_name, "eval-*.parquet")))
        if files:
            # Same explicit decode=False features as the hub path — without
            # them datasets infers a decoding Audio column and demands
            # torchcodec.
            return load_dataset("parquet", data_files=files, split="train",
                                streaming=True, features=_eval_features())
        print(f"    WARN: no local files for {config_name} under {local_dir}; using HF hub")
    return load_dataset(
        GHANA_SPEECH_EVAL,
        config_name,
        split="eval",
        streaming=True,
        features=_eval_features(),
        token=HF_TOKEN or None,
    )


def load_eval_samples(config_name, num_samples=NUM_SAMPLES):
    """Load up to num_samples from a ghana-speech-eval category-config.

    Streams with decode=False, decoding raw audio bytes with soundfile
    (avoids torchcodec). Returns list of dicts:
        [{text, audio, sample_rate, length, language, iso, category}, ...]
    """
    ds = _load_dataset(config_name)

    category = config_name.split("_")[0]
    samples = []
    for row in ds:
        if len(samples) >= num_samples:
            break
        raw = row["audio"]["bytes"]
        if raw is None:
            continue
        audio_array, sr = sf.read(io.BytesIO(raw))
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        samples.append({
            "text": row["text"],
            "audio": audio_array.astype(np.float32),
            "sample_rate": sr,
            "length": row.get("length"),
            "language": row.get("language"),
            "iso": row.get("iso"),
            "category": category,
        })
    return samples
