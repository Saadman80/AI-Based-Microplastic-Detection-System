"""
Download the trained microplastic model FROM Kaggle TO this PC, into ../models/.

This runs on YOUR computer (not on Kaggle) and pulls the model that the training
notebook auto-saved to your Kaggle Dataset (cell 10b: kagglehub.dataset_upload).

── ONE-TIME SETUP ─────────────────────────────────────────────────────────────
1. On kaggle.com: click your avatar → Settings → API → "Create New Token".
   This downloads a file called  kaggle.json  (your personal API key).
2. Put that file here:  C:\\Users\\revon\\.kaggle\\kaggle.json
   (create the .kaggle folder if it doesn't exist)
3. Done. You never touch the token again.

── USAGE ──────────────────────────────────────────────────────────────────────
    python tools/download_trained_model.py
Optional: pass a different dataset handle:
    python tools/download_trained_model.py sadmanmahir/microplastic-yolo26m-model
"""

import os
import sys
import subprocess

DEFAULT_DATASET = "sadmanmahir/microplastic-yolo26m-model"
DEST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def main():
    handle = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET

    # make sure the kaggle library is installed on this PC
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Installing the 'kaggle' library (one-time)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "kaggle"])

    # import here so the pip-install above is in effect
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:  # pragma: no cover
        print("Could not import kaggle:", e)
        return 1

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print("\nAuthentication failed:", e)
        print("Make sure kaggle.json is at  C:\\Users\\revon\\.kaggle\\kaggle.json")
        print("(Download it from kaggle.com -> Settings -> API -> Create New Token.)")
        return 1

    os.makedirs(DEST, exist_ok=True)
    print(f"Downloading  {handle}  ->  {DEST}")
    api.dataset_download_files(handle, path=DEST, unzip=True, quiet=False)
    print("\n[done] Model files are in:", DEST)
    for f in sorted(os.listdir(DEST)):
        print("   ", f)
    print("\nPlastiScope will use models/microplastic_yolo26.onnx automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
