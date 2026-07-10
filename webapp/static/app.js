/* PlastiScope — client. Upload → /api/detect → canvas annotation + live readout. */
"use strict";

const CLASS_COLORS = {
  fiber:    "#D07A4A",
  film:     "#45C2D0",
  fragment: "#B583D6",
  pellet:   "#A7B24E",
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
  img: null,            // HTMLImageElement of current specimen
  filename: null,
  detections: [],       // full list from server (floor 0.05)
  inferenceMs: null,
  threshold: 0.25,
  enabled: new Set(CLASSES),
  busy: false,
};

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

/* ── Health check ──────────────────────────────────────────────────────── */
(async () => {
  try {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error();
    const h = await r.json();
    $("statusLed").classList.add("ok");
    $("statusText").textContent = `${h.model} · ready`;
  } catch {
    $("statusLed").classList.add("err");
    $("statusText").textContent = "model offline";
    showToast("Detection service unreachable. Start the server and reload.", true);
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

/* ── Core: send to model ───────────────────────────────────────────────── */
async function analyzeFile(file) {
  if (state.busy) return;
  if (!file.type.startsWith("image/")) {
    return showToast("That file is not an image. Use JPG, PNG, TIFF, BMP or WEBP.", true);
  }
  state.busy = true;
  state.filename = file.name;

  // Show the specimen immediately, scan while inferring
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

    try {
      const form = new FormData();
      form.append("file", file);
      const r = await fetch("/api/detect", { method: "POST", body: form });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || `Analysis failed (HTTP ${r.status}).`);
      }
      const data = await r.json();
      state.detections = data.detections;
      state.inferenceMs = data.inference_ms;
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

  const lw = Math.max(2, Math.round(w / 320));           // scale with image size
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
