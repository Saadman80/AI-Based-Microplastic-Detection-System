# 🔬 Microplastic Detection System — YOLO26

Deep-learning system that **detects and classifies microplastic particles** in microscopy imagery into four morphological types — **fiber, film, fragment, pellet** — using a custom-trained [Ultralytics **YOLO26**](https://docs.ultralytics.com/models/yolo26/) object detector.

![Task](https://img.shields.io/badge/task-object%20detection-blue)
![Model](https://img.shields.io/badge/model-YOLO26-brightgreen)
![mAP50](https://img.shields.io/badge/test%20mAP@50-0.897-success)
![Framework](https://img.shields.io/badge/framework-PyTorch%20%7C%20Ultralytics-orange)
![License](https://img.shields.io/badge/code-MIT-lightgrey)

> Built end-to-end: sourcing & cleaning four open datasets → harmonizing labels → training on Kaggle GPU → evaluation → export for deployment. A detection **website** is the next milestone.

---

## 📊 Results (held-out test set — 553 images, 3,749 objects)

| Class | AP@50 | Precision | Recall | mAP@50-95 |
|----------|:-----:|:---------:|:------:|:---------:|
| fiber    | 0.935 | 0.883 | 0.907 | 0.688 |
| film     | **0.962** | 0.879 | 0.869 | 0.569 |
| fragment | 0.933 | 0.895 | 0.924 | 0.516 |
| pellet   | 0.757 | 0.778 | 0.777 | 0.347 |
| **all**  | **0.897** | 0.859 | 0.869 | **0.530** |

**Headline: mAP@50 = 0.897** with a lightweight YOLO26-s (9.5 M params, 20.5 GFLOPs, ~11 ms/image inference on a T4). Trained for 135 epochs (early-stopped).

<p align="center">
  <img src="results/sample_predictions.png" width="85%" alt="Sample predictions on test images"><br>
  <em>Model predictions on random test images</em>
</p>

<p align="center">
  <img src="results/results.png" width="49%" alt="Training curves">
  <img src="results/confusion_matrix.png" width="35%" alt="Confusion matrix">
</p>

---

## 🗂️ Dataset

A unified dataset of **8,948 images / 58,026 labeled boxes** was assembled from **four** open microplastic datasets on Roboflow Universe, spanning **two imaging modalities** (brightfield + fluorescence microscopy).

| Split | Images | Boxes |
|-------|:------:|:-----:|
| train | 7,292 | 46,732 |
| valid | 1,103 | 7,545 |
| test  | 553 | 3,749 |

Key data-engineering decisions (full write-up in [`docs/DATASET_REPORT.md`](docs/DATASET_REPORT.md)):
- **Unified 4-class morphological taxonomy** (GESAMP standard). All sources already shared indices 0–2 (fiber/film/fragment); index 3 was harmonized (`pallet`→`pellet`, and `microbeads` folded into the rounded `pellet` class) — achieving a merge with **zero label-file remapping**.
- **Deduplication + split-leak removal** (7 exact duplicates via MD5, train-priority keep).
- **Integrity audited**: 0 corrupt images, 0 orphans, 0 out-of-range boxes.

> The raw images are **not** committed (licensing + size). `data/data.yaml` holds the class map; the report documents exact provenance and how to rebuild.

---

## 📁 Repository structure

```
microplastic_detection_system/
├── data/
│   └── data.yaml            # class names / dataset config
├── models/
│   ├── microplastic_yolo26_best.pt    # trained PyTorch weights
│   └── microplastic_yolo26.onnx       # portable ONNX (NMS-free, web-ready)
├── notebooks/
│   └── microplastic_yolo26_kaggle_training.ipynb   # full training pipeline
├── results/                 # metrics, curves, confusion matrix, sample preds
├── docs/
│   ├── DATASET_REPORT.md            # dataset build & decisions
│   └── KAGGLE_TRAINING_GUIDE.md     # step-by-step training on Kaggle
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick start — run inference

```bash
pip install -r requirements.txt
```

```python
from ultralytics import YOLO

model = YOLO("models/microplastic_yolo26_best.pt")
results = model.predict("your_image.jpg", conf=0.25, imgsz=640)

for b in results[0].boxes:
    cls  = results[0].names[int(b.cls)]      # fiber | film | fragment | pellet
    conf = float(b.conf)
    x1, y1, x2, y2 = b.xyxy[0].tolist()
    print(f"{cls}  {conf:.2f}  [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
```

**Portable / browser deployment:** `models/microplastic_yolo26.onnx`. YOLO26 is **NMS-free**, so the ONNX graph already outputs final decoded detections — ideal for `onnxruntime` (backend) or `onnxruntime-web` (in-browser, no server).

---

## 🏋️ Reproduce the training

Full instructions in [`docs/KAGGLE_TRAINING_GUIDE.md`](docs/KAGGLE_TRAINING_GUIDE.md). In short:
1. Upload the dataset as a Kaggle Dataset.
2. Import `notebooks/microplastic_yolo26_kaggle_training.ipynb`.
3. Accelerator **GPU T4 x2**, Internet **On**, attach the dataset.
4. **Run All** → trains, validates on test, exports `.pt` + `.onnx`, packages outputs.

Config (YOLO26-s, 640 px, batch 16, AMP, early stopping) is in one editable cell.

---

## 🧭 Roadmap
- [x] Dataset sourcing, cleaning, unification & audit
- [x] YOLO26 training on Kaggle GPU + evaluation
- [x] Model export (PyTorch + ONNX)
- [ ] **Detection website** (upload an image → annotated detections + particle counts)
- [ ] Optional accuracy pass (higher-resolution `yolo26m @ 896` to strengthen the `pellet` class)

---

## 🙏 Acknowledgements
Trained on open datasets shared by the community on **Roboflow Universe** (microplastic-detection-fcg6y, microplastic-final-kpdl3, microplastic-6piy9, microplastic-nuga5). Built with [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics). See [`LICENSE`](LICENSE) for data/model usage terms (research/non-commercial due to one CC BY-NC-SA source).

---

<p align="center"><em>Author: Sadman Maher · Computer Vision / Deep Learning portfolio project</em></p>
