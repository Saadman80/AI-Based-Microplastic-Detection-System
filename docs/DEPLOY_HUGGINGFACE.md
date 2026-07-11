# 🚀 Deploy PlastiScope free on Hugging Face Spaces

This publishes the web app at `https://huggingface.co/spaces/<your-username>/plastiscope`.
The Space **clones this GitHub repo** (app + model) and runs it — so you only add **2 small files**.

> **Prerequisite:** this GitHub repo must be **public** (it is, for a portfolio) so the Space can clone it.

---

## Step 1 — Hugging Face account (once)
Sign up (free, no card) at <https://huggingface.co/join> if you don't have one.

## Step 2 — Create the Space
1. Go to <https://huggingface.co/new-space>.
2. **Owner:** you · **Space name:** `plastiscope`.
3. **License:** MIT.
4. **Select the Space SDK:** **Docker** → **Blank**.
5. **Hardware:** **CPU basic** (free) · **Visibility:** Public.
6. Click **Create Space**.

## Step 3 — Add the two files (web UI, no coding)
On your new Space page → **Files** tab → **+ Add file → Create a new file**.

**File 1 — `Dockerfile`** — paste the contents of [`deploy/hf-space/Dockerfile`](../deploy/hf-space/Dockerfile), then **Commit**.

**File 2 — `README.md`** — the Space already has a README; open it → **Edit** → replace everything with the contents of [`deploy/hf-space/README.md`](../deploy/hf-space/README.md) → **Commit**. *(The `---` frontmatter at the top is required — it tells HF to use Docker on port 7860.)*

## Step 4 — Wait for the build
The Space auto-builds (**Building…** badge). It clones the repo, installs the deps, and starts the server — about **2–4 minutes**. Watch **Logs** if you want. When it flips to **Running**, PlastiScope is live. 🎉

Your public URL: `https://huggingface.co/spaces/<your-username>/plastiscope`

---

## Alternative — deploy via git (instead of Step 3)
If you prefer the terminal:
```bash
git clone https://huggingface.co/spaces/<your-username>/plastiscope
cd plastiscope
cp ../AI-Based-Microplastic-Detection-System/deploy/hf-space/Dockerfile .
cp ../AI-Based-Microplastic-Detection-System/deploy/hf-space/README.md .
git add . && git commit -m "PlastiScope on HF Spaces"
git push        # enter your HF username + an access token (Settings → Access Tokens)
```

---

## Updating the demo later
The Space is pinned to whatever the GitHub repo had at build time. After you push new
changes to GitHub, rebuild the Space: Space → **Settings → Factory reboot** (re-clones
the latest code/model).

## Notes
- **First load after inactivity:** free Spaces pause when idle and wake in a few seconds on the next visit.
- **RAM:** free CPU tier is 2 vCPU / 16 GB — plenty for the 78 MB model.
- **Keep the GitHub repo public** for the Space to clone. (If you make it private, switch to uploading the app files + model into the Space directly instead.)
