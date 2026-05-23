"""
download_data.py — Download small, manageable medical datasets via Kaggle API
Datasets used (ALL under 1.5 GB):
  • Brain Tumor MRI     : ~150 MB  (4 classes, 99% accuracy reference notebook)
  • Chest X-ray Pneumonia: ~1.2 GB (Normal vs Pneumonia — NOT the 45GB NIH set)

Usage:
    python download_data.py --dataset brain      # ~150 MB
    python download_data.py --dataset chest      # ~1.2 GB
    python download_data.py --dataset both
    python download_data.py --notebook brain     # pull reference notebook
    python download_data.py --info               # show structure + train commands

Kaggle setup (one-time):
    1. kaggle.com → Profile → Settings → API → Create New Token → kaggle.json
    2. mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
    OR: export KAGGLE_USERNAME=... && export KAGGLE_KEY=...
"""

import os, sys, argparse, subprocess

DATASETS = {
    "brain": {
        "name":     "Brain Tumor MRI (4-Class)",
        "slug":     "masoudnickparvar/brain-tumor-mri-dataset",
        "size":     "~150 MB",
        "dest":     "data/brain_mri",
        "classes":  ["No Tumor", "Glioma", "Meningioma", "Pituitary"],
        "cmd":      "kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset --unzip -p data/brain_mri",
        "note":     "Used in the 99% accuracy reference notebook",
    },
    "chest": {
        "name":     "Chest X-ray Pneumonia (Normal vs Pneumonia)",
        "slug":     "paultimothymooney/chest-xray-pneumonia",
        "size":     "~1.2 GB",
        "dest":     "data/chest_xray",
        "classes":  ["Normal", "Pneumonia"],
        "cmd":      "kaggle datasets download -d paultimothymooney/chest-xray-pneumonia --unzip -p data/chest_xray",
        "note":     "Binary classification — far smaller than NIH 45GB set",
    },
}

NOTEBOOKS = {
    "brain": {
        "name": "Brain Tumor MRI Accuracy 99% (Yousef Mohamed)",
        "slug": "yousefmohamed20/brain-tumor-mri-accuracy-99",
        "cmd":  "kaggle kernels pull yousefmohamed20/brain-tumor-mri-accuracy-99 -p notebooks/",
    },
}


def check_auth() -> bool:
    kj   = os.path.expanduser("~/.kaggle/kaggle.json")
    env  = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if env or os.path.exists(kj):
        return True
    print("""
❌  Kaggle credentials not found.

Quick setup (2 minutes):
  1. Go to https://www.kaggle.com → Your Profile → Settings → API
  2. Click "Create New API Token"  →  downloads kaggle.json
  3. Run:
       mkdir -p ~/.kaggle
       mv ~/Downloads/kaggle.json ~/.kaggle/
       chmod 600 ~/.kaggle/kaggle.json
  4. Re-run this script
""")
    return False


def run(cmd: str, dry: bool) -> bool:
    print(f"  ▶  {cmd}")
    if dry:
        print("     [dry-run — not executed]")
        return True
    result = subprocess.run(cmd.split(), capture_output=False)
    return result.returncode == 0


def download(key: str, dry: bool = False):
    ds = DATASETS[key]
    print(f"\n{'─'*55}")
    print(f"📥  {ds['name']}")
    print(f"    Size    : {ds['size']}")
    print(f"    Classes : {', '.join(ds['classes'])}")
    print(f"    Dest    : {ds['dest']}/")
    print(f"    Note    : {ds['note']}")
    os.makedirs(ds["dest"], exist_ok=True)
    ok = run(ds["cmd"], dry)
    if ok:
        print(f"  ✅  Done → {ds['dest']}/")
    else:
        print(f"  ❌  Failed — check credentials and disk space ({ds['size']} needed)")
    return ok


def pull_notebook(key: str, dry: bool = False):
    nb = NOTEBOOKS[key]
    print(f"\n📓  Pulling notebook: {nb['name']}")
    os.makedirs("notebooks", exist_ok=True)
    ok = run(nb["cmd"], dry)
    if ok:
        print("  ✅  Saved to notebooks/")
    return ok


def show_info():
    print("""
Directory structure after download:
─────────────────────────────────────────────
data/
├── brain_mri/
│   ├── Training/
│   │   ├── glioma/          (~826 images)
│   │   ├── meningioma/      (~822 images)
│   │   ├── notumor/         (~395 images)
│   │   └── pituitary/       (~827 images)
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── chest_xray/
│   ├── train/
│   │   ├── NORMAL/          (~1341 images)
│   │   └── PNEUMONIA/       (~3875 images)
│   ├── val/
│   └── test/
│
└── sample_images/           ← synthetic demo images (pre-included, no download needed)

weights/
├── brain_mri.pth            ← produced by: python train_brain.py
└── chest_xray.pth           ← produced by: python train_chest.py

Training commands (after download):
─────────────────────────────────────────────
python train_brain.py   --epochs 20  # ~15 min on GPU, ~45 min CPU
python train_chest.py   --epochs 10  # ~10 min on GPU, ~30 min CPU
─────────────────────────────────────────────
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download medical imaging datasets (small sizes only)")
    ap.add_argument("--dataset",  choices=["brain","chest","both"])
    ap.add_argument("--notebook", choices=["brain"])
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("--info",     action="store_true")
    args = ap.parse_args()

    print("="*55)
    print("  MedAI — Dataset Downloader  (small datasets only)")
    print("="*55)

    if args.info:
        show_info(); sys.exit(0)

    if not args.dataset and not args.notebook:
        ap.print_help(); print(); show_info(); sys.exit(0)

    if not check_auth() and not args.dry_run:
        sys.exit(1)

    if args.dataset == "both":
        download("brain", args.dry_run)
        download("chest", args.dry_run)
    elif args.dataset:
        download(args.dataset, args.dry_run)

    if args.notebook:
        pull_notebook(args.notebook, args.dry_run)

    show_info()
    print("✅  All done!")
