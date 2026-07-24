---
title: nsanku-ASR Leaderboard
emoji: 🎤
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: true
license: apache-2.0
tags:
  - speech-recognition
  - asr
  - ghana
  - benchmark
  - leaderboard
---

# nsanku-ASR Leaderboard

Benchmarking **organization-owned** ASR models on **43 Ghanaian languages** using the
[ghana-speech-eval](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech-eval) dataset.

Each model is scored on every eval category a language appears in (Bible / JW / Finance /
UNICEF); the reported WER/CER is the **average across categories**. Only models from
organizations (not personal accounts) are included.

Data source: [GhanaNLP/nsanku-ASR](https://github.com/GhanaNLP/nsanku-ASR)