/* PlastiScope — 100% in-browser client.
   Loads the YOLO26-s ONNX model with onnxruntime-web and runs detection on the
   visitor's device (no server). Upload → letterbox → session.run → decode → UI. */
"use strict";

const MODEL_URL = "model/microplastic_yolo26s.onnx";
const IMG_SIZE = 640;
const CONF_FLOOR = 0.05;

const CLASS_COLORS = {
  fiber:    "#B85A32",
  film:     "#2E7FB0",
  fragment: "#8B4FA8",
  pellet:   "#6E7D2E",
};
const CLASSES = Object.keys(CLASS_COLORS);

const $ = (id) => document.getElementById(id);
const stage = $("stage"), emptyState = $("emptyState"), viewport = $("viewport");
const canvas = $("canvas"), ctx = canvas.getContext("2d");
const fileInput = $("fileInput"), dropzone = $("dropzone");
const scanline = $("scanline"), readout = $("readout"), stageMeta = $("stageMeta");
const confSlider = $("confSlider"), confValue = $("confValue");
const toast = $("toast");

const state = {
  img: null,
  filename: null,
  detections: [],
  inferenceMs: null,
  threshold: 0.25,
  enabled: new Set(CLASSES),
  busy: false,
};

let session = null;   // onnxruntime-web InferenceSession

/* ── Eyepiece tick marks (stage micrometer, viewBox 240, center 120) ────── */
(() => {
  const g = document.getElementById("ticks");
  if (!g) return;
  const C = 120;
  for (let i = 0; i < 72; i++) {
    const a = (i / 72) * Math.PI * 2;
    const major = i % 6 === 0;
    const r1 = major ? 100 : 104, r2 = 110;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", C + r1 * Math.cos(a));
    line.setAttribute("y1", C + r1 * Math.sin(a));
    line.setAttribute("x2", C + r2 * Math.cos(a));
    line.setAttribute("y2", C + r2 * Math.sin(a));
    g.appendChild(line);
  }
})();

/* ── Load the model in the browser (once) ──────────────────────────────── */
(async () => {
  const statusText = $("statusText"), led = $("statusLed");
  try {
    statusText.textContent = "loading model…";
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
    led.classList.add("ok");
    statusText.textContent = "YOLO26-s · ready";
  } catch (e) {
    led.classList.add("err");
    statusText.textContent = "model failed to load";
    showToast("Could not load the detection model: " + e.message, true);
  }
})();

/* ── Upload paths: click, drop, paste, samples ─────────────────────────── */
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) analyzeFile(fileInput.files[0]);
});

["dragover", "dragenter"].forEach((ev) =>
  stage.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) =>
  stage.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
stage.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) analyzeFile(f);
});

document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) analyzeFile(item.getAsFile());
});

document.querySelectorAll(".sample-chip").forEach((chip) =>
  chip.addEventListener("click", async () => {
    const r = await fetch(chip.dataset.src);
    const blob = await r.blob();
    analyzeFile(new File([blob], chip.dataset.src.split("/").pop(), { type: blob.type }));
  }));

$("newSpecimen").addEventListener("click", resetStage);

/* ── Preprocess: letterbox to 640×640, NCHW float32 /255 ────────────────── */
function preprocess(img) {
  const size = IMG_SIZE;
  const w0 = img.naturalWidth, h0 = img.naturalHeight;
  const s = Math.min(size / w0, size / h0);
  const nw = Math.round(w0 * s), nh = Math.round(h0 * s);
  const px = Math.floor((size - nw) / 2), py = Math.floor((size - nh) / 2);

  const c = document.createElement("canvas");
  c.width = size; c.height = size;
  const cx = c.getContext("2d");
  cx.fillStyle = "rgb(114,114,114)";
  cx.fillRect(0, 0, size, size);
  cx.drawImage(img, px, py, nw, nh);

  const { data } = cx.getImageData(0, 0, size, size);   // RGBA
  const plane = size * size;
  const arr = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    arr[i]             = data[i * 4]     / 255;   // R
    arr[plane + i]     = data[i * 4 + 1] / 255;   // G
    arr[2 * plane + i] = data[i * 4 + 2] / 255;   // B
  }
  return { tensor: new ort.Tensor("float32", arr, [1, 3, size, size]), s, px, py, w0, h0 };
}

/* ── Decode: (300×6) letterboxed [x1,y1,x2,y2,score,cls] → image pixels ─── */
function decode(out, s, px, py, w0, h0) {
  const dets = [];
  for (let i = 0; i < 300; i++) {
    const o = i * 6;
    const score = out[o + 4];
    if (score < CONF_FLOOR) continue;
    const cid = Math.round(out[o + 5]);
    if (cid < 0 || cid >= CLASSES.length) continue;
    let x1 = Math.max(0, Math.min((out[o]     - px) / s, w0));
    let y1 = Math.max(0, Math.min((out[o + 1] - py) / s, h0));
    let x2 = Math.max(0, Math.min((out[o + 2] - px) / s, w0));
    let y2 = Math.max(0, Math.min((out[o + 3] - py) / s, h0));
    if (x2 - x1 < 1 || y2 - y1 < 1) continue;
    dets.push({
      class_id: cid,
      class_name: CLASSES[cid],
      confidence: Math.round(score * 10000) / 10000,
      box: [x1, y1, x2, y2].map((v) => Math.round(v * 10) / 10),
    });
  }
  dets.sort((a, b) => b.confidence - a.confidence);
  return dets;
}

/* ── Core: run the model in-browser ────────────────────────────────────── */
async function analyzeFile(file) {
  if (state.busy) return;
  if (!file.type.startsWith("image/")) {
    return showToast("That file is not an image. Use JPG, PNG, TIFF, BMP or WEBP.", true);
  }
  state.busy = true;
  state.filename = file.name;

  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = async () => {
    state.img = img;
    emptyState.hidden = true;
    viewport.hidden = false;
    $("newSpecimen").hidden = false;
    scanline.hidden = false;
    stageMeta.textContent = `${file.name} · ${img.naturalWidth}×${img.naturalHeight}px`;
    readout.textContent = "analyzing…";
    drawScene();
    await new Promise((r) => setTimeout(r, 20));   // let the UI paint before the blocking run

    try {
      if (!session) throw new Error("Model still loading — try again in a moment.");
      const t0 = performance.now();
      const { tensor, s, px, py, w0, h0 } = preprocess(img);
      const feeds = {}; feeds[session.inputNames[0]] = tensor;
      const results = await session.run(feeds);
      const out = results[session.outputNames[0]].data;
      state.detections = decode(out, s, px, py, w0, h0);
      state.inferenceMs = Math.round(performance.now() - t0);
      refresh();
    } catch (err) {
      showToast(err.message, true);
      readout.textContent = "analysis failed";
    } finally {
      scanline.hidden = true;
      state.busy = false;
      URL.revokeObjectURL(url);
    }
  };
  img.onerror = () => {
    state.busy = false;
    showToast("Could not read that image file.", true);
  };
  img.src = url;
}

function resetStage() {
  state.img = null;
  state.detections = [];
  emptyState.hidden = false;
  viewport.hidden = true;
  $("newSpecimen").hidden = true;
  stageMeta.textContent = "";
  readout.textContent = "—";
  fileInput.value = "";
  CLASSES.forEach((c) => (document.querySelector(`[data-count="${c}"]`).textContent = "–"));
  $("totalCount").textContent = "–";
  ["exportPng", "exportCsv", "exportJson"].forEach((id) => ($(id).disabled = true));
}

/* ── Filtering + rendering ─────────────────────────────────────────────── */
function visibleDetections() {
  return state.detections.filter(
    (d) => d.confidence >= state.threshold && state.enabled.has(d.class_name)
  );
}

function refresh() {
  const vis = visibleDetections();
  const counts = Object.fromEntries(CLASSES.map((c) => [c, 0]));
  vis.forEach((d) => counts[d.class_name]++);

  CLASSES.forEach((c) => {
    document.querySelector(`[data-count="${c}"]`).textContent = state.enabled.has(c) ? counts[c] : "·";
  });
  $("totalCount").textContent = vis.length;

  readout.textContent =
    `${vis.length} particle${vis.length === 1 ? "" : "s"} ≥ ${state.threshold.toFixed(2)} · ` +
    `inference ${state.inferenceMs} ms`;

  ["exportPng", "exportCsv", "exportJson"].forEach((id) => ($(id).disabled = vis.length === 0 && !state.img));
  drawScene(vis);
}

function drawScene(vis = visibleDetections()) {
  if (!state.img) return;
  const { naturalWidth: w, naturalHeight: h } = state.img;
  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(state.img, 0, 0);

  const lw = Math.max(2, Math.round(w / 320));
  const fs = Math.max(11, Math.round(w / 46));
  ctx.font = `600 ${fs}px "Spline Sans Mono", monospace`;
  ctx.textBaseline = "top";

  for (const d of vis) {
    const [x1, y1, x2, y2] = d.box;
    const color = CLASS_COLORS[d.class_name] || "#fff";
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${d.class_name} ${(d.confidence * 100).toFixed(0)}`;
    const tw = ctx.measureText(label).width;
    const ly = y1 - fs - lw * 2 >= 0 ? y1 - fs - lw * 2 : y1 + lw;
    ctx.fillStyle = color;
    ctx.fillRect(x1 - lw / 2, ly, tw + fs * 0.7, fs + lw * 1.6);
    ctx.fillStyle = "#141414";
    ctx.fillText(label, x1 + fs * 0.3, ly + lw * 0.9);
  }
}

/* ── Controls ──────────────────────────────────────────────────────────── */
confSlider.addEventListener("input", () => {
  state.threshold = parseFloat(confSlider.value);
  confValue.textContent = state.threshold.toFixed(2);
  if (state.img) refresh();
});

document.querySelectorAll(".ch-toggle").forEach((btn) => {
  btn.addEventListener("click", () => {
    const cls = btn.closest(".channel").dataset.class;
    const on = btn.getAttribute("aria-pressed") === "true";
    btn.setAttribute("aria-pressed", String(!on));
    on ? state.enabled.delete(cls) : state.enabled.add(cls);
    if (state.img) refresh();
  });
});

/* ── Exports ───────────────────────────────────────────────────────────── */
function downloadBlob(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}
const stem = () => (state.filename || "specimen").replace(/\.[^.]+$/, "");

$("exportPng").addEventListener("click", () =>
  canvas.toBlob((b) => downloadBlob(b, `${stem()}_annotated.png`), "image/png"));

$("exportCsv").addEventListener("click", () => {
  const rows = [["class", "confidence", "x1", "y1", "x2", "y2"]];
  visibleDetections().forEach((d) => rows.push([d.class_name, d.confidence, ...d.box]));
  const csv = rows.map((r) => r.join(",")).join("\n");
  downloadBlob(new Blob([csv], { type: "text/csv" }), `${stem()}_detections.csv`);
});

$("exportJson").addEventListener("click", () => {
  const payload = {
    filename: state.filename,
    threshold: state.threshold,
    inference_ms: state.inferenceMs,
    detections: visibleDetections(),
  };
  downloadBlob(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
    `${stem()}_detections.json`);
});

/* ── Toast ─────────────────────────────────────────────────────────────── */
let toastTimer;
function showToast(msg, isError = false) {
  toast.textContent = msg;
  toast.classList.toggle("err", isError);
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (toast.hidden = true), 4200);
}
