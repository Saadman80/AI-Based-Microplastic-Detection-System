---
title: PlastiScope Microplastic Detection
emoji: 🔬
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🔬 PlastiScope — Microplastic Detection

Instrument-style web app that detects and classifies microplastics —
**fiber · film · fragment · pellet** — in microscopy images using a custom-trained
**YOLO26m** model, served with **FastAPI + ONNX Runtime**.

Drop a micrograph on the microscope "stage" to get annotated detections, per-class
particle counts, a live confidence slider, and CSV / JSON / image export.

**Source code & full project:** https://github.com/Saadman80/AI-Based-Microplastic-Detection-System

> This Space builds by cloning the GitHub repository above (which contains the app
> and the trained model) and running the FastAPI server on port 7860.
