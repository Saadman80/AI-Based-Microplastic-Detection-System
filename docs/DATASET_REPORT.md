# Microplastic Detection — Unified YOLO26 Dataset Report

**Prepared:** 2026-07-09
**Output location:** `C:\Users\revon\Desktop\microplastic_YOLO26_final`
**Total size:** ~330 MB · **8,948 images** · **58,026 bounding boxes** · **4 classes**

---

## 1. Source datasets

Four Roboflow YOLO exports were consolidated:

| Tag | Source folder | License | Images | Imaging modality |
|-----|---------------|---------|-------:|------------------|
| ds1 | microplastic detection.v1i.yolo26 | CC BY 4.0 | 1,596 | Brightfield microscopy |
| ds2 | microplastic-final.v2-main-2.yolo26 | CC BY 4.0 | 956 | Brightfield microscopy |
| ds3 | Microplastic.v5i.yolo26 | CC BY 4.0 | 6,003 | **Fluorescence** microscopy |
| ds4 | MicroPlastic.v7-relabeling.yolo26 | **BY-NC-SA 4.0** | 400 | Brightfield microscopy |

> ⚠️ **License note:** ds4 is **BY-NC-SA 4.0 (non-commercial, share-alike)**. Because it is included, the merged dataset as a whole inherits **non-commercial** terms. If you intend to commercialize the website/model, either obtain permission for ds4 or exclude ds4 (it is only 400 images / 4.5%, so exclusion has minimal impact).

## 2. Audit findings (pre-merge)

A full integrity audit was run on all 12 source splits:

- **Corrupt images:** 0
- **Orphans** (image without label / label without image): 0
- **Malformed boxes / out-of-range coordinates:** 0
- **All images:** JPEG. Sizes: 480×480 (ds1) and 640×640 (ds2/ds3/ds4).
- **Empty label files (background/negative images):** 143, all in ds3 — kept intentionally (they help suppress false positives).
- **Exact-duplicate image groups:** 7 (identified by MD5). Of these, several were **train↔valid split-leaks** — same image present in two splits, which inflates validation metrics.

The source data was exceptionally clean; the only substantive issues were taxonomy harmonization, the 7 duplicates, and the license/modality considerations documented here.

## 3. Key decisions

### 3.1 Class taxonomy (why no label remapping was needed)
All four datasets already use identical class **indices** for the first three morphologies:

| Index | ds1 | ds2 | ds3 | ds4 | → Unified |
|------:|-----|-----|-----|-----|-----------|
| 0 | `0 fiber` | fiber | Fibre | fiber | **fiber** |
| 1 | `1 film` | film | Film | film | **film** |
| 2 | `2 fragment` | fragment | Fragment | fragment | **fragment** |
| 3 | `3 pallet` | pellet | microbeads | pallet | **pellet** |

- Names were only inconsistent in **casing/spelling** (`pallet` is a misspelling of `pellet`).
- **Index 3** was `pellet` in 3 datasets and `microbeads` in ds3. Both are the **rounded/spherical morphology** class in the standard microplastic taxonomy, so they were consolidated into a single `pellet` class. Because the index was already `3` everywhere, **no label file had to be edited** — this is the safest possible merge (zero risk of remap errors).

Result: a clean 4-class morphological taxonomy — `fiber, film, fragment, pellet` — aligned with the GESAMP / Free et al. classification used in microplastics research.

### 3.2 Two imaging modalities — kept together
ds3 is **fluorescence** microscopy (dark field, stained/glowing particles); ds1/ds2/ds4 are **brightfield** microscopy (filter membrane, light background). Both modalities were retained so the model generalizes to real-world microscope inputs of either type. Both appear in **train and validation**, so evaluation is representative. If you later want a purely brightfield model, exclude ds3; for a fluorescence-only model, use ds3 alone.

### 3.3 Deduplication & split-leak removal
For each duplicate group, exactly one copy was kept using the priority **train > valid > test** (so a leaked image is retained only in train, never in val/test). 7 redundant copies were dropped. No leakage remains.

### 3.4 Split policy
Each source dataset's original train/valid/test assignment was **preserved** (no re-shuffling), which avoids accidentally moving near-duplicate augmentations across the split boundary.

## 4. Final dataset

### Structure
```
microplastic_YOLO26_final/
├── data.yaml
├── DATASET_REPORT.md
├── train/  images/ (7,292)  labels/ (7,292)
├── valid/  images/ (1,103)  labels/ (1,103)
└── test/   images/ (  553)  labels/ (  553)
```
Every filename is prefixed with its source tag (`ds1_`…`ds4_`) to guarantee uniqueness and preserve provenance.

### Split sizes
| Split | Images | % | Boxes | Background imgs |
|-------|-------:|--:|------:|----------------:|
| train | 7,292 | 81.5% | 46,732 | 121 |
| valid | 1,103 | 12.3% | 7,545 | 17 |
| test  |   553 |  6.2% | 3,749 | 5 |
| **Total** | **8,948** | 100% | **58,026** | 143 |

### Class distribution (bounding boxes)
| Class | Total | train | valid | test |
|-------|------:|------:|------:|-----:|
| fiber    | 19,756 | 15,908 | 2,584 | 1,264 |
| film     |  9,273 |  7,417 | 1,229 |   627 |
| fragment | 10,494 |  8,402 | 1,368 |   724 |
| pellet   | 18,503 | 15,005 | 2,364 | 1,134 |

Balance is healthy: the rarest class (`film`) to most common (`fiber`) ratio is ~1:2.1 — well within what YOLO handles without special weighting.

## 5. Training recommendations

1. **Model:** start with `yolo26s` (small) or `yolo26m` (medium) at `imgsz=640`. The dataset mixes 480 and 640 px; training at 640 lets Ultralytics letterbox the 480 px images cleanly.
2. **Command:**
   ```bash
   yolo detect train data="C:/Users/revon/Desktop/microplastic_YOLO26_final/data.yaml" \
        model=yolo26s.pt epochs=150 imgsz=640 batch=16 patience=30
   ```
3. **Augmentation:** keep default mosaic/mixup but consider `hsv_v`/`hsv_s` moderate — colour carries morphology signal (fibers are often dark, pellets coloured). Avoid vertical/horizontal flips being disabled; flips are fine here.
4. **Evaluate per class** — watch `film` (rarest) and cross-modality performance. Report mAP50 and mAP50-95 on the held-out `test` split.
5. **If commercial use is required,** rebuild excluding ds4 (see §1 license note); the provided scripts make this a one-line change.

## 6. Reproducibility
Processing scripts (`audit.py`, `build.py`, `verify.py`) are available in the working scratchpad. The pipeline is deterministic: audit → taxonomy harmonization → dedup(train>valid>test priority) → provenance-prefixed copy → verification.
