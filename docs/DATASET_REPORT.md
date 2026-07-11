# Microplastic Detection — YOLO26 Training Dataset Report

**Prepared:** 2026-07-11
**Dataset:** `microplastic_yolo26_training_dataset`
**Total:** **9,238 images · ~58,945 bounding boxes · 4 classes** (fiber · film · fragment · pellet)

---

## 1. Source datasets

The dataset is built around an **originally-collected in-house microplastic set prepared at the Micro and Nano (MINA) Laboratory, University of Dhaka.** At MINA, microplastic specimens were imaged under a microscope and every particle was hand-annotated by the author — this in-house set (290 brightfield micrographs) is the real-world core of the dataset and the reason the model is grounded in genuine laboratory conditions.

To broaden morphological variety and cross-instrument robustness, the MINA set was combined with **four openly-shared community microplastic datasets from Roboflow Universe**, spanning both brightfield and fluorescence microscopy. These are gratefully credited as contributing sources: `microplastic-detection-fcg6y`, `microplastic-final-kpdl3`, `microplastic-6piy9`, and `microplastic-nuga5`.

| Source | Images | Imaging modality |
|--------|-------:|------------------|
| **University of Dhaka — MINA (Micro & Nano) Laboratory** *(in-house, author-collected & labeled)* | 290 | Brightfield microscopy |
| Community datasets (4 combined, Roboflow Universe) | 8,948 | Brightfield + fluorescence |
| **Total** | **9,238** | |

## 2. Audit findings

Every source split passed a full integrity audit before merging:

- **Corrupt images:** 0  ·  **Orphans** (image without label or vice-versa): 0  ·  **Malformed / out-of-range boxes:** 0
- **All JPEG.** Sizes: 480×480, 512×512 (MINA lab), and 640×640.
- **Background/negative images** (empty label files): ~155, kept intentionally — they help suppress false positives.
- **Exact duplicates (MD5):** 7 in the community data (several were train↔valid split-leaks) + 1 within the MINA set. All resolved.
- The MINA set had **0 duplicates against the community data** — it is entirely novel imagery.

## 3. Key decisions

### 3.1 Class taxonomy (no label remapping needed)
Every source — including the MINA lab set — already uses identical class **indices** for all four morphologies:

| Index | MINA lab | community sources | → Unified |
|------:|----------|-------------------|-----------|
| 0 | Fiber | fiber / Fibre / `0 fiber` | **fiber** |
| 1 | Flim* | film / Film / `1 film` | **film** |
| 2 | Fragment | fragment / Fragment | **fragment** |
| 3 | Pellet | pellet / microbeads / pallet | **pellet** |

\*Only cosmetic naming differences existed (casing / the `Flim`, `pallet` typos); label files use the index, so **not a single label had to be rewritten** — the safest possible merge. Index 3 consolidates the rounded/spherical morphology (`pellet` + `microbeads`) into one `pellet` class, matching the GESAMP / Free et al. morphological taxonomy used in microplastics research.

### 3.2 Two imaging modalities kept together
The community data includes **fluorescence** microscopy (dark field, stained particles) alongside **brightfield** (filter membrane, light background); the MINA set is brightfield. All modalities appear in **train and validation**, so the model generalises across microscope types and evaluation stays representative.

### 3.3 Merging the MINA lab set
The 290 MINA micrographs were split **80/10/10** into train/valid/test and given a `lab_` filename prefix (community images keep `ds1_`…`ds4_`). The `lab_` prefix means lab-domain performance can be measured on its own at any time — important for verifying real-world accuracy, not just aggregate scores.

### 3.4 Deduplication & split-leak removal
For every duplicate group, exactly one copy was kept with priority **train > valid > test**, so no image leaks across splits (which would inflate validation metrics).

## 4. Final dataset

### Structure
```
microplastic_yolo26_training_dataset/
├── data.yaml
├── DATASET_REPORT.md
├── train/  images/ (7,524)  labels/ (7,524)
├── valid/  images/ (1,132)  labels/ (1,132)
└── test/   images/ (  582)  labels/ (  582)
```
Filenames are prefixed by source (`lab_`, `ds1_`…`ds4_`) to guarantee uniqueness and preserve provenance.

### Split sizes
| Split | Images | of which MINA lab | Boxes | Background imgs |
|-------|-------:|------------------:|------:|----------------:|
| train | 7,524 | 232 | 47,495 | 129 |
| valid | 1,132 | 29 | ~7,646 | 19 |
| test  | 582 | 29 | ~3,804 | ~7 |
| **Total** | **9,238** | **290** | **~58,945** | ~155 |

### Class distribution (bounding boxes)
| Class | Total | train | valid | test |
|-------|------:|------:|------:|-----:|
| fiber    | 19,765 | 15,914 | 2,585 | 1,266 |
| film     |  9,468 |  7,576 | 1,254 |   638 |
| fragment | 11,077 |  8,894 | 1,431 |   752 |
| pellet   | 18,635 | 15,111 | 2,376 | 1,148 |

Balance is healthy (rarest `film` : most-common `fiber` ≈ 1 : 2.1) — well within what YOLO handles without special class weighting.

## 5. Training recommendations

Training is driven by the notebook at [`notebooks/microplastic_yolo26_kaggle_training.ipynb`](../notebooks/microplastic_yolo26_kaggle_training.ipynb).

1. **Model:** `yolo26m` at `imgsz=640` (the dataset mixes 480/512/640 px; 640 letterboxes them cleanly). For a further accuracy push on small particles, `imgsz=768` is the highest-impact knob — see the notebook's config notes.
2. **Hardware:** Kaggle **GPU T4 x2** with DDP (`device=[0,1]`, `batch=32`). Run via **Save & Run All (Commit)** so outputs persist (an interactive session is wiped on idle timeout).
3. **Augmentation:** strengthened for lighting / colour-cast / orientation robustness (`hsv_h/s/v` raised, `degrees=15`, `flipud=0.5`, `mixup=0.15`, `close_mosaic=15`) — improves generalisation to varied microscope conditions while holding in-domain accuracy.
4. **Evaluate per class and per source** — report mAP50 / mAP50-95 on the held-out `test` split, and separately on the `lab_` images to confirm real MINA-domain performance.

## 6. Reproducibility
Deterministic pipeline (`seed=42`): audit → taxonomy harmonisation → dedup (train>valid>test priority) → provenance-prefixed copy + 80/10/10 lab split → verification. The class map lives in `data.yaml`.
