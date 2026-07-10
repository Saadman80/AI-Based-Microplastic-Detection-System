# 🚀 Kaggle Training Guide — Microplastic YOLO26

**Goal:** train your YOLO26 microplastic detector on Kaggle's free GPU and get a model you can plug into your website.

**You have two files on your Desktop:**
| File | What it is | Where it goes |
|------|-----------|---------------|
| `microplastic_YOLO26_final.zip` (307 MB) | Your prepared dataset | Upload as a **Kaggle Dataset** |
| `microplastic_yolo26_kaggle_training.ipynb` | The full training notebook (22 cells) | Import as a **Kaggle Notebook** |

---

## 1. Which accelerator? → **GPU T4 x2** (not P100, not TPU)

Pick a **GPU** over the TPU (Ultralytics YOLO runs on PyTorch/CUDA; the TPU needs an XLA rewrite Ultralytics doesn't have). Among the GPUs, use **T4**, **not P100**.

| | **GPU T4 x2** ✅ | **GPU P100** ⚠️ | **TPU v5e-8** ❌ |
|---|---|---|---|
| Supported by Kaggle's current PyTorch (2.10+cu128)? | **Yes** (Turing, sm_75) | **No** — Pascal (sm_60) kernels were dropped → `no kernel image` error | N/A |
| Effort | Zero — just select it | Must downgrade torch to `2.5.1+cu121` + restart | Days of XLA work |
| VRAM | 16 GB (×2 cards) | 16 GB | — |
| Verdict | **Use this** | Only if T4 unavailable (see below) | Wrong tool for YOLO |

> **Why not P100 anymore?** Kaggle upgraded its base image to **PyTorch 2.10 (CUDA 12.8)**, whose binaries **no longer include Pascal (P100) kernels**. Any attempt to move the model to a P100 fails with *"CUDA error: no kernel image is available for execution on the device."* The T4 (Turing) is fully supported and is actually a bit faster here thanks to Tensor Cores. The notebook uses one T4 (`device=0`); you can set `device=[0,1]` in cell 5 to use both, though single-GPU is simpler and reliable in notebooks.
>
> **Forced onto P100?** Replace cell 2 with:
> ```python
> import torch
> !pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
> !pip install -q -U ultralytics --no-deps
> !pip install -q ultralytics-thop py-cpuinfo
> ```
> then **Kernel → Restart** and Run All. Slower to set up (~2 GB download); T4 is the better path.

---

## 2. Step-by-step

### Step A — Upload the dataset (once)
1. Go to **kaggle.com → Datasets → New Dataset**.
2. Drag in **`microplastic_YOLO26_final.zip`**. Kaggle auto-extracts it.
3. Title it e.g. **`microplastic-yolo26`**. Set visibility (Private is fine). Click **Create**.
4. Wait for it to finish processing (a few minutes for 307 MB).

### Step B — Create the notebook
1. **kaggle.com → Code → New Notebook**.
2. **File → Import Notebook** → upload **`microplastic_yolo26_kaggle_training.ipynb`**.

### Step C — Configure the session (right-hand panel)
1. **Accelerator → GPU T4 x2.** *(Not P100 — see §1; Kaggle's current PyTorch doesn't support P100.)*
2. **Internet → On.** *(required — the notebook pip-installs ultralytics and downloads pretrained weights)*
3. **Input → + Add Input → Datasets →** search your `microplastic-yolo26` dataset → **Add**.

### Step D — Run
- Click **Run All** (or Ctrl+F9). Watch cell 1 confirm the P100 and cell 3 confirm it found your images.
- The notebook auto-detects the dataset path, writes a Kaggle-correct `data.yaml`, trains, validates on the test split, exports, and packages everything.

### Step E — Get your model
- Open the **Output** tab (or `/kaggle/working`) and download **`microplastic_model_package.zip`**. It contains:
  - `microplastic_yolo26_best.pt` — PyTorch weights (best epoch)
  - `microplastic_yolo26.onnx` — portable model for web/backend
  - `results.png`, `confusion_matrix.png`, `PR_curve.png`, `results.csv` — training curves & metrics
  - `sample_predictions.png` — model predictions on test images
  - `data.yaml`, `args.yaml` — configs

> **Tip for unattended runs:** instead of interactive Run All, use **Save Version → Save & Run All (Commit)**. It runs headless for up to 12 h and preserves all outputs even if you close the tab.

---

## 3. What the notebook does (cell map)

1. **GPU check** — verifies P100 + prints `nvidia-smi`.
2. **Install** — `pip -U ultralytics` (brings YOLO26).
3. **Auto-detect dataset** — finds `*/train/images` under `/kaggle/input` (robust to zip layout).
4. **Write `data.yaml`** — regenerates it with the Kaggle mount path (your uploaded one has a Windows path).
5. **Config** — model size, epochs, image size, batch — all in one editable cell.
6. **Train** — YOLO26-s, 640px, AMP, early stopping, plots on. Has a `RESUME` switch.
7. **Validate on TEST** — mAP50, mAP50-95, per-class AP.
8. **Export** — copies `best.pt`, exports **ONNX** (YOLO26 is NMS-free → decoded boxes straight out).
9. **Sample inference** — draws predictions on 6 random test images.
10. **Package** — bundles everything into one downloadable zip.
11. **Website usage** — code snippets (see §6).

---

## 4. Expected time, tuning & metrics

- **Training time:** YOLO26-s @ 640px, batch 16, ~7,300 train images on P100 ≈ **2–2.5 min/epoch → ~5–6 h for 150 epochs**. Early stopping (`patience=40`) usually finishes sooner. Well within the 12 h limit.
- **Want more accuracy?** In cell 5 set `MODEL = 'yolo26m.pt'` (slower, higher mAP). For a lighter/faster web model use `'yolo26n.pt'`.
- **Out-of-memory?** Lower `BATCH` to 8. **Under-utilised GPU?** Raise `BATCH` to 32.
- **Realistic expectation:** with ~58k labeled boxes across 4 balanced classes and clean data, a solid **mAP50 in the ~0.75–0.90 range** is typical for this kind of microscopy detection — your test-split numbers in cell 7 are the honest measure.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| Cell 3: "Could not find */train/images" | You didn't attach the dataset. Right panel → Add Input → your dataset. |
| Cell 2/6: download/install errors | **Internet is Off.** Turn it On in settings, restart & Run All. |
| `CUDA error: no kernel image is available for execution on the device` | You're on **P100**, which Kaggle's PyTorch 2.10+cu128 no longer supports. **Switch Accelerator → GPU T4 x2** and Run All. (Or, to stay on P100, use the torch 2.5.1+cu121 downgrade in §1.) |
| `CUDA out of memory` | Lower `BATCH` (16→8) in cell 5. |
| `WARNING ... cache not writeable` | Harmless — `/kaggle/input` is read-only; training proceeds normally. |
| Session hit 12 h | Use **Save & Run All (Commit)**, or reduce `EPOCHS`; early stopping usually converges earlier. |
| YOLO26 weight name error | The notebook auto-falls back to `yolo11s.pt`; or `pip install -U ultralytics` is out of date — re-run cell 2. |

---

## 6. Using the trained model in your website

**Option A — Python backend (recommended: most accurate, least code).** FastAPI/Flask:
```python
from ultralytics import YOLO
model = YOLO('microplastic_yolo26_best.pt')

def detect(image_path):
    r = model.predict(image_path, conf=0.25, imgsz=640)[0]
    return [{
        'class': r.names[int(b.cls)],
        'confidence': round(float(b.conf), 3),
        'bbox': [round(v) for v in b.xyxy[0].tolist()],  # x1,y1,x2,y2
    } for b in r.boxes]
```

**Option B — In-browser, no server (`onnxruntime-web`).** Ship `microplastic_yolo26.onnx` with your frontend. Because YOLO26 is **NMS-free**, the ONNX output is already the final detections — you just:
1. letterbox the input image to 640×640, normalize `/255`, layout NCHW float32;
2. run the session;
3. filter boxes by confidence (e.g. ≥0.25) and scale coords back to original size.
Classes (index order): `['fiber', 'film', 'fragment', 'pellet']`.

**Option C — ONNX in a Python backend (no ultralytics dependency).** Same as B but with `onnxruntime` server-side — smallest dependency footprint for production.

---

## 7. Class reference
`0 = fiber · 1 = film · 2 = fragment · 3 = pellet`
(`pellet` includes rounded/spherical particles, i.e. the merged pellet+microbeads morphology.)

---

*Dataset details and merge decisions are documented in `microplastic_YOLO26_final/DATASET_REPORT.md`.*
