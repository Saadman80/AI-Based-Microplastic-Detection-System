# 🚀 Deploy PlastiScope free on Hugging Face Spaces (Gradio)

Publishes the demo at `https://huggingface.co/spaces/<your-username>/plastiscope` — **free**.

> **Why Gradio, not Docker?** Hugging Face now puts the **Docker** SDK behind a *paid*
> plan. The **Gradio** SDK is free (CPU basic) and perfect for serving an ONNX model.
> The app runs server-side (~0.3 s/image); the model is fetched from this GitHub repo
> at startup, so the Space needs only **2 small files**.

---

## Step 1 — Hugging Face account (once)
Sign up (free, no card) at <https://huggingface.co/join>.

## Step 2 — Create the Space
1. Go to <https://huggingface.co/new-space>.
2. **Space name:** `plastiscope` · **License:** MIT.
3. **Select the Space SDK → Gradio → Blank.**
4. **Hardware:** **CPU basic** (free) · **Visibility:** Public → **Create Space**.

## Step 3 — Add the two files (web UI, no coding)
On the Space → **Files** tab → **+ Add file → Create a new file**:

**File 1 — `app.py`** — paste the contents of [`deploy/hf-gradio/app.py`](../deploy/hf-gradio/app.py) → **Commit**.

**File 2 — `requirements.txt`** — paste [`deploy/hf-gradio/requirements.txt`](../deploy/hf-gradio/requirements.txt) → **Commit**.

*(The Space's README already has the right settings if you created it as Gradio. To be safe, you can also replace the README with [`deploy/hf-gradio/README.md`](../deploy/hf-gradio/README.md) — its `---` frontmatter pins `sdk: gradio`, `app_file: app.py`.)*

## Step 4 — Wait for the build
The Space installs the deps, downloads the model from GitHub, and launches Gradio —
about **2–4 minutes** (**Building… → Running**). Then it's live. 🎉

Your public URL: `https://huggingface.co/spaces/<your-username>/plastiscope`

---

## Alternative — deploy via git
```bash
git clone https://huggingface.co/spaces/<your-username>/plastiscope
cd plastiscope
cp ../AI-Based-Microplastic-Detection-System/deploy/hf-gradio/* .
git add . && git commit -m "PlastiScope (Gradio)"
git push        # enter your HF username + an access token (Settings → Access Tokens)
```

## Notes
- **First load after idle:** free Spaces pause when idle and wake in a few seconds.
- **Model:** `microplastic_yolo26m.onnx` is fetched from the public GitHub repo at startup — keep the repo **public**.
- **Updating:** after pushing changes to GitHub, restart the Space (**Settings → Factory reboot**) to re-fetch.
- A **Docker** version (identical FastAPI app, if you ever get a paid plan) is kept in [`deploy/hf-space/`](../deploy/hf-space/).
