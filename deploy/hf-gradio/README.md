---
title: PlastiScope Microplastic Detection
emoji: 🔬
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 🔬 PlastiScope — Microplastic Detection

Detect and classify microplastics — **fiber · film · fragment · pellet** — in
microscopy images with a custom-trained **YOLO26m** model (ONNX Runtime, CPU).

Upload a micrograph (or try a sample), tune the confidence threshold, and get
annotated detections plus per-class particle counts.

**Full project & source:** https://github.com/Saadman80/AI-Based-Microplastic-Detection-System

> The model (`microplastic_yolo26m.onnx`) is fetched from the GitHub repo above at
> startup, so this Space needs only `app.py` + `requirements.txt`.
