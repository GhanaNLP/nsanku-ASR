"""nsanku-ASR Leaderboard — HuggingFace Space.

Fetches benchmark results from the GhanaNLP/nsanku-ASR GitHub repo and displays
interactive leaderboards for organization-owned ASR models on Ghanaian languages.

Scoring: each model is evaluated on every eval category a language appears in
(bible / jw / finance / unicef) and the reported WER/CER is the **average across
those categories**.
"""

import io
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
    "kdh": "Tem", "bov": "Tuwuli", "vag": "Vagla", "gaa": "Ga", "aha": "Ahanta",
}

CATEGORY_LABELS = {
    "bible": "Bible", "jw": "JW", "finance": "Finance", "unicef": "UNICEF",
}


def _categorize_model(model_id):
    m = model_id.lower()
    if "faster-whisper" in m:
        return "Faster-Whisper"
    if "whisper" in m:
        return "Whisper"
    if "w2v-bert" in m or "w2v_bert" in m or "wav2vec2-bert" in m:
        return "Wav2Vec2-BERT"
    if "mms" in m:
        return "MMS"
    if "wav2vec" in m or "w2v2" in m or "w2v-" in m:
        return "wav2vec2"
    if "xls-r" in m or "xlsr" in m:
        return "XLS-R"
    if "hubert" in m:
        return "HuBERT"
    if "seamless" in m:
        return "SeamlessM4T"
    if "simba" in m:
        return "Simba"
    if "xeus" in m or "espnet" in m:
        return "ESPnet"
    return "Other"


def _classify_error(error_str):
    if not error_str:
        return ""
    e = str(error_str).lower()
    if "gated" in e or "403" in e:
        return "Gated repo"
    if "cudnn" in e:
        return "cuDNN error"
    if "architecture_not_supported" in e or ("does not support" in e and "attention" in e):
        return "Attn not supported"
    if "unknown_architecture" in e or "unrecognized" in e:
        return "Unknown architecture"
    if "no_valid_output" in e:
        return "No valid output"
    if "load_failed" in e:
        return "Load failed"
    return str(error_str)[:40]


def _fetch_yaml(path):
    r = requests.get(f"{RAW_BASE}/{path}", timeout=30)
    return yaml.safe_load(r.text) if r.status_code == 200 else None


def _list_benchmark_files():
    r = requests.get(f"{API_BASE}/benchmarks", timeout=30)
    if r.status_code != 200:
        return []
    return [f["name"] for f in r.json()
            if f["name"].endswith(".yaml") and not f["name"].startswith("_")]


def _fmt_categories(per_category):
    """Return 'bible 71.2 / unicef 59.2' style WER breakdown."""
    if not per_category:
        return ""
    parts = []
    for cat, v in per_category.items():
        w = v.get("wer")
        label = CATEGORY_LABELS.get(cat, cat)
        parts.append(f"{label} {w * 100:.1f}" if w is not None else f"{label} —")
    return "  |  ".join(parts)


def load_all_data():
    rows = []
    for fname in sorted(_list_benchmark_files()):
        iso = fname.replace(".yaml", "")
        data = _fetch_yaml(f"benchmarks/{fname}")
        if not data or "benchmarks" not in data:
            continue
        lang_name = data.get("language") or LANG_NAMES.get(iso, iso)
        categories = data.get("categories", [])
        n = data.get("num_samples_per_category", 300)
        for b in data["benchmarks"]:
            wer = b.get("wer")
            cer = b.get("cer")
            model = b.get("model", "?")
            per_cat = b.get("per_category", {})
            rows.append({
                "iso": iso,
                "language": lang_name,
                "categories": ", ".join(CATEGORY_LABELS.get(c, c) for c in categories),
                "model": model,
                "owner": b.get("owner", model.split("/")[0]),
                "track": b.get("model_class") or ("llm" if "gemini" in model.lower() else "non-llm"),
                "model_type": _categorize_model(model),
                "params": b.get("params", "?"),
                "wer": round(wer * 100, 2) if wer is not None else None,
                "cer": round(cer * 100, 2) if cer is not None else None,
                "per_category_wer": _fmt_categories(per_cat),
                "n_categories": len(per_cat) if per_cat else 0,
                "status": "pass" if wer is not None else "fail",
                "fail_reason": _classify_error(b.get("error")),
                "url": b.get("model_url", f"https://huggingface.co/{model}"),
                "num_samples": n,
            })
    return pd.DataFrame(rows)


def build_global_leaderboard(df):
    """Best org model per language — ranked by averaged WER."""
    passed = df[df["status"] == "pass"].copy()
    if passed.empty:
        return pd.DataFrame()
    best = passed.loc[passed.groupby("iso")["wer"].idxmin()].sort_values("wer")
    best.insert(0, "rank", range(1, len(best) + 1))
    cols = ["rank", "language", "categories", "model", "owner", "track", "model_type",
            "params", "wer", "cer"]
    return best[cols].reset_index(drop=True)


def build_per_language(df, iso, model_type_filter="All", sort_by="wer"):
    sub = df[df["iso"] == iso].copy()
    if model_type_filter != "All":
        sub = sub[sub["model_type"] == model_type_filter]
    sub = sub.sort_values(sort_by if sort_by in sub.columns else "wer", na_position="last")
    sub.insert(0, "rank", range(1, len(sub) + 1))
    passed = sub[sub["status"] == "pass"]
    failed = sub[sub["status"] == "fail"]
    cols_pass = ["rank", "model", "owner", "track", "model_type", "params", "wer", "cer",
                 "per_category_wer"]
    cols_fail = ["rank", "model", "owner", "track", "model_type", "fail_reason"]
    return passed[cols_pass].reset_index(drop=True), failed[cols_fail].reset_index(drop=True)


def build_model_status(df, status_filter="All", lang_filter="All", search=""):
    sub = df.copy()
    if status_filter != "All":
        sub = sub[sub["status"] == status_filter]
    if lang_filter != "All":
        sub = sub[sub["language"] == lang_filter]
    if search:
        sub = sub[sub["model"].str.contains(search, case=False, na=False)]
    cols = ["language", "model", "owner", "track", "model_type", "params", "wer", "cer",
            "status", "fail_reason"]
    return sub[cols].sort_values(["language", "wer"], na_position="last").reset_index(drop=True)


def build_header(df):
    total_langs = df["iso"].nunique()
    total_models = df["model"].nunique()
    total_orgs = df["owner"].nunique()
    passed = df[df["status"] == "pass"]
    best = passed.nsmallest(1, "wer")
    best_str = "—"
    if not best.empty:
        r = best.iloc[0]
        best_str = f"{r['model']} ({r['language']}) — {r['wer']:.2f}% WER"
    return f"""
| Languages | Org models | Organizations | Evaluations | Best result |
|---|---|---|---|---|
| **{total_langs}** | **{total_models}** | **{total_orgs}** | **{len(df)}** | **{best_str}** |
"""


with gr.Blocks(
    title="nsanku-ASR Leaderboard",
    theme=gr.themes.Soft(),
    css=".main { max-width: 1200px; margin: auto; } h1 { text-align: center; }",
) as app:
    gr.Markdown("""
    # nsanku-ASR Leaderboard

    Benchmarking **organization-owned** ASR models on **Ghanaian languages** using the
    [ghana-speech-eval](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech-eval) dataset.

    Each model is scored on every eval category a language appears in
    (**Bible / JW / Finance / UNICEF**); the reported **WER/CER is the average across categories**.
    Only models from organizations (not personal accounts) are included.

    Data: [GhanaNLP/nsanku-ASR](https://github.com/GhanaNLP/nsanku-ASR)
    """)

    df_state = gr.State()
    header_md = gr.Markdown()

    with gr.Row():
        refresh_btn = gr.Button("Load / Refresh Data", variant="primary", size="lg")

    def _load():
        df = load_all_data()
        header = build_header(df)
        global_lb = build_global_leaderboard(df)
        lang_choices = ["All"] + sorted(df["language"].unique().tolist())
        type_choices = ["All"] + sorted(df["model_type"].unique().tolist())
        first_lang = lang_choices[1] if len(lang_choices) > 1 else "All"
        return (df, header, global_lb,
                gr.update(choices=lang_choices, value=first_lang),
                gr.update(choices=type_choices),
                gr.update(choices=lang_choices))

    with gr.Tab("Global Leaderboard"):
        gr.Markdown("**Best org model per language** — ranked by average WER (lower is better).")
        global_df = gr.DataFrame()

    with gr.Tab("Per-Language"):
        with gr.Row():
            lang_dd = gr.Dropdown(label="Language", choices=[], scale=2)
            type_dd_lang = gr.Dropdown(label="Model Type", choices=["All"], value="All", scale=1)
            sort_dd = gr.Dropdown(label="Sort by", choices=["wer", "cer"], value="wer", scale=1)
        gr.Markdown("### Evaluated models  \n*`per_category_wer` shows the WER within each category (%).*")
        pass_df = gr.DataFrame()
        gr.Markdown("### Failed to evaluate")
        fail_df = gr.DataFrame()

        def _update_lang(df, iso_or_name, mtype, sortby):
            # dropdown holds language names; map back to iso
            iso = iso_or_name
            if df is not None and iso_or_name not in df["iso"].values:
                match = df[df["language"] == iso_or_name]
                iso = match["iso"].iloc[0] if not match.empty else iso_or_name
            return build_per_language(df, iso, mtype, sortby)

        for trigger in [lang_dd, type_dd_lang, sort_dd]:
            trigger.change(_update_lang,
                           inputs=[df_state, lang_dd, type_dd_lang, sort_dd],
                           outputs=[pass_df, fail_df])

    with gr.Tab("Model Status"):
        with gr.Row():
            status_dd = gr.Dropdown(label="Status", choices=["All", "pass", "fail"], value="All", scale=1)
            lang_dd_status = gr.Dropdown(label="Language", choices=["All"], value="All", scale=1)
            search_box = gr.Textbox(label="Search model name", placeholder="whisper, mms, w2v-bert…", scale=2)
        status_table = gr.DataFrame()

        def _update_status(df, status, lang, search):
            return build_model_status(df, status, lang, search)

        for trigger in [status_dd, lang_dd_status, search_box]:
            trigger.change(_update_status,
                           inputs=[df_state, status_dd, lang_dd_status, search_box],
                           outputs=[status_table])

    gr.Markdown("""
    ---
    **WER** = Word Error Rate · **CER** = Character Error Rate · lower is better · shown as percentages.

    Final score = mean of the per-category WER/CER for the categories each language has.
    Some models fail to load (gated repos, CTranslate2/ESPnet formats, unsupported architectures, cuDNN on Hopper).
    """)

    refresh_btn.click(
        _load,
        outputs=[df_state, header_md, global_df, lang_dd, type_dd_lang, lang_dd_status],
    ).then(
        _update_lang,
        inputs=[df_state, lang_dd, type_dd_lang, sort_dd],
        outputs=[pass_df, fail_df],
    ).then(
        _update_status,
        inputs=[df_state, status_dd, lang_dd_status, search_box],
        outputs=[status_table],
    )


if __name__ == "__main__":
    app.launch()
