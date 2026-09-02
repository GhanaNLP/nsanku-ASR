---
title: nsanku ASR Benchmark
emoji: 🎤
colorFrom: blue
colorTo: indigo
sdk: static
pinned: true
license: apache-2.0
tags:
  - speech-recognition
  - asr
  - ghana
  - benchmark
  - leaderboard
---

# nsanku ASR Benchmark

Benchmarking ASR models on **Ghanaian languages** using the
[ghana-speech-eval](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech-eval) dataset.

Three tracks: **ASR (open models)** (downloadable weights anyone can re-run), **ASR (closed models)**
(proprietary hosted APIs) and **LLM**.
Scored per category (Bible / JW / Finance / Unicef / LDS / WAXAL), averaged as the final WER/CER.
**Ranking uses the average of WER and CER** — every model row links to a recipe
(`code` badge) showing the exact inference code used to run it, so model authors
can correct how their model is evaluated via a pull request.

Data source: [GhanaNLP/nsanku-ASR](https://github.com/GhanaNLP/nsanku-ASR)
