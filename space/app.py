"""nsanku-ASR Leaderboard — HuggingFace Space.

Fetches benchmark results from the GhanaNLP/nsanku-ASR GitHub repo
and displays interactive leaderboards for ASR models on Ghanaian languages.
"""

import io
import csv
import requests
import pandas as pd
import yaml
import gradio as gr

REPO = "GhanaNLP/nsanku-ASR"
BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
API_BASE = f"https://api.github.com/repos/{REPO}/contents"

LANG_NAMES = {
    "twi": "Twi (Akuapem + Asante)", "ewe": "Ewe", "hau": "Hausa",
    "kbp": "Kabiye", "dag": "Dagbani", "dga": "Dagaare",
    "fat": "Fante", "nko": "Nkonya", "any": "Anyin", "avn": "Avatime",
    "bud": "Bassar Ntcham", "bim": "Bimoba", "biv": "Birifor Southern",
    "bib": "Bissa", "bwu": "Buli", "ncu": "Chumburung", "ada": "Dangme",
    "mzw": "Deg", "ffm": "Fulfulde Maasina", "acd": "Gikyode",
    "gjn": "Gonja", "xsm": "Kasem", "xon": "Konkomba", "kma": "Konni",
    "kus": "Kusaal", "lef": "Lelemi", "maw": "Mampruli", "naw": "Nawuri",
    "gur": "Ninkare", "ntr": "Ntrubo", "nzi": "Nzema", "sig": "Paasaal",
    "sfw": "Sehwi", "lip": "Sekpele", "snw": "Selee",
    "sil": "Sisaala Tumulung", "akp": "Siwu", "tpm": "Tampulma",
    "kdh": "Tem", "bov": "Tuwuli", "vag": "Vagla",
}


def _categorize_model(model_id):
    """Return a human-readable model type."""
    m = model_id.lower()
    if "gemini" in m:
        return "Gemini API"
    if "whisper" in m:
        if "faster-whisper" in m:
            return "Faster-Whisper"
        return "Whisper"
    if "mms" in m:
        return "MMS"
    if "wav2vec" in m or "w2v" in m:
        return "wav2vec2"
    if "xls-r" in m or "xlsr" in m:
        return "XLS-R"
    if "hubert" in m:
        return "HuBERT"
    if "simba" in m:
        return "SeamlessM4T (Simba)"
    if "seamless" in m:
        return "SeamlessM4T"
    if "xeus" in m or "espnet" in m:
        return "ESPnet"
    if "zeroswot" in m:
        return "ZeroSwot"
    if "heep" in m:
        return "HEEP"
    return "Other"


def _classify_error(error_str):
    """Classify failure reason into a short label."""
    if not error_str:
        return "Unknown"
    e = error_str.lower()
    if "gated" in e or "403" in e:
        return "Gated repo"
    if "cudnn" in e:
        return "cuDNN error"
    if "outdated" in e and "generation config" in e:
        return "Outdated gen config"
    if "unsupported language" in e:
        return "Language not supported"
    if "unrecognized" in e and "class" in e:
        return "Unknown architecture"
    if "does not support" in e and "attention" in e:
        return "Attn not supported"
    if "text input must be" in e:
        return "Tokenizer error"
    if "failed to load" in e:
        return "Load failed"
    return error_str[:40]


def _fetch_yaml(path):
    """Fetch a YAML file from the repo."""
    url = f"{RAW_BASE}/{path}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    return yaml.safe_load(r.text)


def _fetch_csv(path):
    """Fetch a CSV file from the repo."""
    url = f"{RAW_BASE}/{path}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    return pd.read_csv(io.StringIO(r.text))


def _list_benchmark_files():
    """List all YAML files in benchmarks/ directory."""
    r = requests.get(f"{API_BASE}/benchmarks", timeout=30)
    if r.status_code != 200:
        return []
    files = r.json()
    return [f["name"] for f in files if f["name"].endswith(".yaml") and not f["name"].startswith("_")]


def load_all_data():
    """Load all benchmark data and model status from GitHub."""
    yaml_files = _list_benchmark_files()

    all_rows = []
    for fname in sorted(yaml_files):
        iso = fname.replace(".yaml", "")
        data = _fetch_yaml(f"benchmarks/{fname}")
        if not data or "benchmarks" not in data:
            continue

        lang_name = LANG_NAMES.get(iso, iso)
        num_samples = data.get("num_samples", 300)

        for b in data["benchmarks"]:
            model = b.get("model", "?")
            wer = b.get("wer")
            cer = b.get("cer")
            error = b.get("error")

            all_rows.append({
                "iso": iso,
                "language": lang_name,
                "model": model,
                "model_type": _categorize_model(model),
                "params": b.get("params", "?"),
                "wer": round(wer * 100, 2) if wer is not None else None,
                "cer": round(cer * 100, 2) if cer is not None else None,
                "speed": b.get("avg_seconds_per_sample"),
                "status": "pass" if wer is not None else "fail",
                "fail_reason": _classify_error(error) if error else "",
                "url": b.get("model_url", f"https://huggingface.co/{model}"),
                "num_samples": num_samples,
            })

    df = pd.DataFrame(all_rows)

    status_df = _fetch_csv("model_status.csv")
    if status_df is None:
        status_df = df.copy()

    return df, status_df


def build_global_leaderboard(df):
    """Best model per language — sorted by WER."""
    passed = df[df["status"] == "pass"].copy()
    if passed.empty:
        return pd.DataFrame()

    idx = passed.groupby("iso")["wer"].idxmin()
    best = passed.loc[idx].sort_values("wer")

    best.insert(0, "rank", range(1, len(best) + 1))
    cols = ["rank", "language", "model", "model_type", "params", "wer", "cer", "speed"]
    return best[cols].reset_index(drop=True)


def build_per_language(df, iso, model_type_filter="All", sort_by="wer"):
    """Leaderboard for a specific language."""
    sub = df[df["iso"] == iso].copy()
    if model_type_filter != "All":
        sub = sub[sub["model_type"] == model_type_filter]

    sub = sub.sort_values(sort_by if sort_by in sub.columns else "wer", na_last=True)
    sub.insert(0, "rank", range(1, len(sub) + 1))

    show_passed = sub[sub["status"] == "pass"]
    show_failed = sub[sub["status"] == "fail"]

    cols_pass = ["rank", "model", "model_type", "params", "wer", "cer", "speed"]
    cols_fail = ["rank", "model", "model_type", "params", "fail_reason"]

    return show_passed[cols_pass].reset_index(drop=True), show_failed[cols_fail].reset_index(drop=True)


def build_model_status(df, status_filter="All", lang_filter="All", search=""):
    """Model × language status table."""
    sub = df.copy()
    if status_filter != "All":
        sub = sub[sub["status"] == status_filter]
    if lang_filter != "All":
        sub = sub[sub["language"] == lang_filter]
    if search:
        sub = sub[sub["model"].str.contains(search, case=False, na=False)]

    cols = ["language", "model", "model_type", "params", "wer", "cer", "status", "fail_reason"]
    return sub[cols].sort_values(["language", "wer"], na_position="last").reset_index(drop=True)


def build_summary(df):
    """Summary stats."""
    total_langs = df["iso"].nunique()
    total_models = df["model"].nunique()
    total_evals = len(df)
    passed = len(df[df["status"] == "pass"])
    failed = len(df[df["status"] == "fail"])
    pass_rate = f"{passed / total_evals * 100:.1f}%" if total_evals else "—"

    gemini = df[(df["model_type"] == "Gemini API") & (df["status"] == "pass")]
    avg_gemini_wer = f"{gemini['wer'].mean():.2f}%" if not gemini.empty else "—"

    best_overall = df[df["status"] == "pass"].nsmallest(1, "wer")
    if not best_overall.empty:
        row = best_overall.iloc[0]
        best_str = f"{row['model']} ({row['language']}) — {row['wer']:.2f}%"
    else:
        best_str = "—"

    md = f"""
## Overview

| Metric | Value |
|---|---|
| Languages evaluated | **{total_langs}** |
| Unique models tested | **{total_models}** |
| Total evaluations | **{total_evals}** |
| Passed | **{passed}** ({pass_rate}) |
| Failed | **{failed}** |
| Avg Gemini WER (all langs) | **{avg_gemini_wer}** |
| Best overall | **{best_str}** |
"""
    return md


def build_model_type_chart(df):
    """Pass/fail counts by model type."""
    sub = df.copy()
    ct = sub.groupby(["model_type", "status"]).size().unstack(fill_value=0)
    if "pass" not in ct.columns:
        ct["pass"] = 0
    if "fail" not in ct.columns:
        ct["fail"] = 0
    ct = ct.sort_values("pass", ascending=False)
    return ct.reset_index()


with gr.Blocks(
    title="nsanku-ASR Leaderboard",
    theme=gr.themes.Soft(),
    css="""
    .main { max-width: 1200px; margin: auto; }
    h1 { text-align: center; }
    """,
) as app:
    gr.Markdown("""
    # nsanku-ASR Leaderboard

    Benchmarking ASR models on **41 Ghanaian language varieties** using the
    [ghana-speech](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech) dataset.

    Data: [GhanaNLP/nsanku-ASR](https://github.com/GhanaNLP/nsanku-ASR) | 300 samples per language
    """)

    df_state = gr.State()
    status_df_state = gr.State()

    with gr.Row():
        refresh_btn = gr.Button("Load / Refresh Data", variant="primary", size="lg")

    def _load():
        df, sdf = load_all_data()
        summary = build_summary(df)
        global_lb = build_global_leaderboard(df)
        lang_choices = ["All"] + sorted(df["language"].unique().tolist())
        type_choices = ["All"] + sorted(df["model_type"].unique().tolist())
        return df, sdf, summary, global_lb, gr.update(choices=lang_choices, value=lang_choices[1] if len(lang_choices) > 1 else "All"), gr.update(choices=type_choices)

    with gr.Tab("Overview"):
        summary_md = gr.Markdown()
        gr.Markdown("### Pass/Fail by Model Type")
        type_chart = gr.DataFrame()

    with gr.Tab("Global Leaderboard"):
        gr.Markdown("**Best model per language** — sorted by WER (lower is better)")
        global_df = gr.DataFrame()

    with gr.Tab("Per-Language"):
        with gr.Row():
            lang_dd = gr.Dropdown(label="Language", choices=[], scale=2)
            type_dd_lang = gr.Dropdown(label="Model Type", choices=["All"], scale=1)
        with gr.Row():
            sort_dd = gr.Dropdown(label="Sort by", choices=["wer", "cer", "speed"], value="wer", scale=1)
        gr.Markdown("### Passed Models")
        pass_df = gr.DataFrame()
        gr.Markdown("### Failed Models")
        fail_df = gr.DataFrame()

        def _update_lang(df, iso, mtype, sortby):
            lang_name = LANG_NAMES.get(iso, iso) if iso != "All" else "All"
            p, f = build_per_language(df, iso, mtype, sortby)
            return p, f

        for trigger in [lang_dd, type_dd_lang, sort_dd]:
            trigger.change(
                _update_lang,
                inputs=[df_state, lang_dd, type_dd_lang, sort_dd],
                outputs=[pass_df, fail_df],
            )

    with gr.Tab("Model Status"):
        with gr.Row():
            status_dd = gr.Dropdown(label="Status", choices=["All", "pass", "fail"], value="All", scale=1)
            lang_dd_status = gr.Dropdown(label="Language", choices=["All"], scale=1)
            search_box = gr.Textbox(label="Search model name", placeholder="e.g. whisper, mms, gemini...", scale=2)
        status_table = gr.DataFrame()

        def _update_status(df, status, lang, search):
            return build_model_status(df, status, lang, search)

        for trigger in [status_dd, lang_dd_status, search_box]:
            trigger.change(
                _update_status,
                inputs=[df_state, status_dd, lang_dd_status, search_box],
                outputs=[status_table],
            )

    gr.Markdown("""
    ---
    **WER** = Word Error Rate | **CER** = Character Error Rate | Lower is better.

    Values shown as percentages. Speed = avg seconds per sample.

    Some models failed due to: gated repos, cuDNN compatibility (H200 Hopper), outdated generation configs, or unsupported architectures.
    """)

    refresh_btn.click(
        _load,
        outputs=[df_state, status_df_state, summary_md, global_df, lang_dd, type_dd_lang],
    ).then(
        _update_lang,
        inputs=[df_state, lang_dd, type_dd_lang, sort_dd],
        outputs=[pass_df, fail_df],
    ).then(
        _update_status,
        inputs=[df_state, status_dd, lang_dd_status, search_box],
        outputs=[status_table],
    ).then(
        lambda df: build_model_type_chart(df),
        inputs=[df_state],
        outputs=[type_chart],
    )


if __name__ == "__main__":
    app.launch()