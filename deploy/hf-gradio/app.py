"""
PlastiScope — microplastic detection, Gradio app for a free Hugging Face Space.

Loads the custom-trained YOLO26m ONNX model (fetched from the public GitHub repo)
and serves an upload → annotated-detections + per-class counts interface.
Runs server-side on the free CPU tier (no PyTorch, ~0.3 s/image).
"""

import os
import numpy as np
import onnxruntime as ort
import requests
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# ── Config ──────────────────────────────────────────────────────────────────
REPO = "https://raw.githubusercontent.com/Saadman80/AI-Based-Microplastic-Detection-System/main"
MODEL_URL = f"{REPO}/models/microplastic_yolo26m.onnx"
MODEL_PATH = "microplastic_yolo26m.onnx"
SAMPLE_FILES = ["brightfield_1.jpg", "brightfield_2.jpg", "fluorescence_1.jpg", "fluorescence_2.jpg"]

IMG_SIZE = 640
CLASS_NAMES = ["fiber", "film", "fragment", "pellet"]
CLASS_COLORS = {
    "fiber": (184, 90, 50), "film": (46, 127, 176),
    "fragment": (139, 79, 168), "pellet": (110, 125, 46),
}


def _download(url, path):
    if not os.path.exists(path):
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


# ── Load model (once, at startup) ───────────────────────────────────────────
_download(MODEL_URL, MODEL_PATH)
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name

# Pre-fetch sample specimens for the Examples gallery (best-effort)
EXAMPLES = []
os.makedirs("samples", exist_ok=True)
for fn in SAMPLE_FILES:
    try:
        p = _download(f"{REPO}/webapp/static/samples/{fn}", f"samples/{fn}")
        EXAMPLES.append([p, 0.25])
    except Exception:
        pass


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def letterbox(img, size=IMG_SIZE):
    w0, h0 = img.size
    s = min(size / w0, size / h0)
    nw, nh = round(w0 * s), round(h0 * s)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None]), s, px, py


def detect(image, conf=0.25):
    """image: RGB numpy array from Gradio. Returns (annotated PIL image, counts dict)."""
    if image is None:
        return None, {"info": "Upload a micrograph to begin."}

    img = Image.fromarray(image).convert("RGB")
    w0, h0 = img.size
    tensor, s, px, py = letterbox(img)
    raw = session.run(None, {INPUT_NAME: tensor})[0][0]  # (300, 6): x1,y1,x2,y2,score,cls

    draw = ImageDraw.Draw(img)
    lw = max(2, round(max(w0, h0) / 320))
    font = _font(max(14, round(max(w0, h0) / 45)))
    counts = {c: 0 for c in CLASS_NAMES}

    for x1, y1, x2, y2, score, cid in raw:
        if score < conf:
            continue
        ci = int(cid)
        if not 0 <= ci < len(CLASS_NAMES):
            continue
        name = CLASS_NAMES[ci]
        col = CLASS_COLORS[name]
        bx1 = max(0.0, (x1 - px) / s)
        by1 = max(0.0, (y1 - py) / s)
        bx2 = min(float(w0), (x2 - px) / s)
        by2 = min(float(h0), (y2 - py) / s)
        if bx2 - bx1 < 1 or by2 - by1 < 1:
            continue

        draw.rectangle([bx1, by1, bx2, by2], outline=col, width=lw)
        label = f"{name} {int(score * 100)}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = by1 - th - 4 if by1 - th - 4 >= 0 else by1
        draw.rectangle([bx1, ly, bx1 + tw + 6, ly + th + 4], fill=col)
        draw.text((bx1 + 3, ly + 2), label, fill=(255, 255, 255), font=font)
        counts[name] += 1

    result = {k: counts[k] for k in CLASS_NAMES}
    result["TOTAL"] = sum(counts.values())
    return img, result


# ── UI ──────────────────────────────────────────────────────────────────────
DESC = (
    "Detect and classify microplastics — **fiber · film · fragment · pellet** — in "
    "microscopy images with a custom-trained **YOLO26m** model.\n\n"
    "Upload a micrograph (or try a sample), adjust the confidence threshold, and hit **Detect**."
)

with gr.Blocks(title="PlastiScope — Microplastic Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 PlastiScope — Microplastic Detection")
    gr.Markdown(DESC)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="Micrograph", type="numpy")
            conf = gr.Slider(0.05, 0.90, value=0.25, step=0.01, label="Confidence threshold")
            btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Detections")
            out_counts = gr.JSON(label="Particle counts")
    if EXAMPLES:
        gr.Examples(examples=EXAMPLES, inputs=[inp, conf], label="Sample specimens")
    gr.Markdown(
        "Model: YOLO26m · test mAP@50 0.895 · trained on 9,238 micrographs "
        "(MINA Lab, Univ. of Dhaka + community data). "
        "[Source](https://github.com/Saadman80/AI-Based-Microplastic-Detection-System)"
    )
    btn.click(detect, inputs=[inp, conf], outputs=[out_img, out_counts])

if __name__ == "__main__":
    demo.launch()
