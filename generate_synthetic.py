"""
generate_synthetic.py — Synthetic Medical Image Generator
Creates realistic-looking training images so you can run/demo the full pipeline
without downloading any dataset.

Usage:
    python generate_synthetic.py             # generate all
    python generate_synthetic.py --type brain
    python generate_synthetic.py --type chest
    python generate_synthetic.py --samples 200   # images per class
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import os, argparse, math, random
from pathlib import Path

random.seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# BASE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def _noise(shape, mu=0, sigma=12):
    return np.random.normal(mu, sigma, shape)

def _save(arr, path):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr).convert("RGB")
    else:
        img = Image.fromarray(arr)
    img.save(path)


# ─────────────────────────────────────────────────────────────────────────────
# BRAIN MRI GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def _brain_base(size=224) -> np.ndarray:
    arr = np.zeros((size, size), dtype=float)
    cx, cy = size // 2, size // 2
    # Skull ring
    for y in range(size):
        for x in range(size):
            d = math.sqrt((x-cx)**2 + (y-cy)**2)
            r = size * 0.45
            if r - 10 < d < r + 4:
                arr[y, x] = random.uniform(180, 220)
            elif d < r - 10:
                arr[y, x] = random.uniform(55, 90)
    # White matter core
    for y in range(size):
        for x in range(size):
            d = math.sqrt((x-cx)**2 + (y-cy)**2)
            if d < size * 0.28:
                arr[y, x] = random.uniform(95, 130)
    # Ventricles
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx-18, cy-25, cx-2, cy+25], fill=random.randint(10, 25))
    draw.ellipse([cx+2,  cy-25, cx+18, cy+25], fill=random.randint(10, 25))
    arr = np.array(img, dtype=float)
    return arr


def brain_notumor(path, size=224):
    arr = _brain_base(size) + _noise((size, size), 0, 10)
    _save(arr, path)


def brain_glioma(path, size=224):
    arr = _brain_base(size)
    cx, cy = size // 2, size // 2
    # Irregular bright mass
    angle = random.uniform(0, 2*math.pi)
    tx = int(cx + random.uniform(25, 55) * math.cos(angle))
    ty = int(cy + random.uniform(25, 55) * math.sin(angle))
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    rw, rh = random.randint(22, 38), random.randint(18, 32)
    draw.ellipse([tx-rw, ty-rh, tx+rw, ty+rh], fill=random.randint(160, 210))
    # Necrotic core
    draw.ellipse([tx-rw//3, ty-rh//3, tx+rw//3, ty+rh//3],
                 fill=random.randint(20, 50))
    arr = np.array(img, dtype=float) + _noise((size, size), 0, 12)
    _save(arr, path)


def brain_meningioma(path, size=224):
    arr = _brain_base(size)
    cx, cy = size // 2, size // 2
    # Surface-attached bright lesion
    angle = random.uniform(0, 2*math.pi)
    r = size * 0.38
    tx = int(cx + r * math.cos(angle))
    ty = int(cy + r * math.sin(angle))
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    rw, rh = random.randint(16, 28), random.randint(14, 24)
    draw.ellipse([tx-rw, ty-rh, tx+rw, ty+rh], fill=random.randint(170, 220))
    arr = np.array(img, dtype=float) + _noise((size, size), 0, 10)
    _save(arr, path)


def brain_pituitary(path, size=224):
    arr = _brain_base(size)
    cx, cy = size // 2, size // 2
    # Central bright spot (pituitary region)
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    py = cy + int(size * 0.12)
    draw.ellipse([cx-14, py-10, cx+14, py+10], fill=random.randint(180, 230))
    arr = np.array(img, dtype=float) + _noise((size, size), 0, 8)
    _save(arr, path)


# ─────────────────────────────────────────────────────────────────────────────
# CHEST X-RAY GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def _chest_base(size=224) -> np.ndarray:
    arr = np.full((size, size), 15, dtype=float)
    cx, cy = size // 2, size // 2
    # Lung fields
    for y in range(size):
        for x in range(size):
            # Left lung
            lx, ly = size*0.32, size*0.50
            if ((x-lx*0.6)**2/(lx**2) + (y-ly)**2/(ly**2)) < 1:
                arr[y, x] = random.uniform(50, 80)
            # Right lung
            rx, ry = size*0.32, size*0.50
            if ((x-(size-lx*0.6))**2/(rx**2) + (y-ry)**2/(ry**2)) < 1:
                arr[y, x] = random.uniform(50, 80)
    # Heart
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx-size//10, size//3, cx+size//8, size*2//3], fill=45)
    # Ribs
    for i in range(6):
        y = size//5 + i*(size//9)
        draw.arc([size//8, y, size*5//12, y+size//12], 200, 340, fill=130, width=2)
        draw.arc([size*7//12, y, size*7//8, y+size//12], 200, 340, fill=130, width=2)
    # Spine
    for i in range(10):
        draw.rectangle([cx-8, size//7+i*size//12, cx+8, size//7+i*size//12+size//14],
                       fill=random.randint(160, 200))
    arr = np.array(img, dtype=float)
    return arr


def chest_normal(path, size=224):
    arr = _chest_base(size) + _noise((size, size), 0, 8)
    _save(arr, path)


def chest_pneumonia(path, size=224):
    arr = _chest_base(size)
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    # Consolidation patches (white-ish areas in lung fields)
    n_patches = random.randint(1, 3)
    cx, cy = size // 2, size // 2
    for _ in range(n_patches):
        side = random.choice([-1, 1])
        px = int(cx + side * random.uniform(size*0.1, size*0.28))
        py = int(cy + random.uniform(-size*0.1, size*0.15))
        rw = random.randint(size//10, size//6)
        rh = random.randint(size//12, size//8)
        draw.ellipse([px-rw, py-rh, px+rw, py+rh],
                     fill=random.randint(120, 180))
    # Air bronchograms
    for _ in range(random.randint(2, 4)):
        px = int(cx + random.uniform(-size*0.25, size*0.25))
        py = int(cy + random.uniform(-size*0.05, size*0.2))
        draw.line([(px, py-12), (px, py+12)], fill=30, width=2)
    arr = np.array(img, dtype=float) + _noise((size, size), 0, 10)
    _save(arr, path)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def _augment(pil_img: Image.Image) -> Image.Image:
    # Random horizontal flip
    if random.random() > 0.5:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random rotation ±15°
    angle = random.uniform(-15, 15)
    pil_img = pil_img.rotate(angle, fillcolor=(0, 0, 0))
    # Random brightness
    pil_img = ImageEnhance.Brightness(pil_img).enhance(random.uniform(0.8, 1.2))
    # Random contrast
    pil_img = ImageEnhance.Contrast(pil_img).enhance(random.uniform(0.85, 1.15))
    # Slight blur
    if random.random() > 0.7:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.9)))
    return pil_img


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────
BRAIN_GEN = {
    "notumor":    brain_notumor,
    "glioma":     brain_glioma,
    "meningioma": brain_meningioma,
    "pituitary":  brain_pituitary,
}

CHEST_GEN = {
    "NORMAL":    chest_normal,
    "PNEUMONIA": chest_pneumonia,
}


def generate_brain(n_per_class=150, size=224, base="data/brain_mri"):
    print(f"\n🧠 Generating Brain MRI dataset ({n_per_class} images/class) …")
    splits = {"Training": int(n_per_class * 0.85), "Testing": n_per_class - int(n_per_class * 0.85)}
    total = 0
    for split, n in splits.items():
        for cls, gen_fn in BRAIN_GEN.items():
            out = Path(base) / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                path = str(out / f"{cls}_{i:04d}.png")
                tmp  = str(out / f"_tmp_{i}.png")
                gen_fn(tmp, size)
                img = Image.open(tmp)
                if i > 0:  # keep first as-is, augment rest
                    img = _augment(img)
                img.save(path)
                os.remove(tmp)
                total += 1
            print(f"   {split}/{cls}: {n} images")
    print(f"✅ Brain MRI: {total} images → {base}/")
    return total


def generate_chest(n_per_class=200, size=224, base="data/chest_xray"):
    print(f"\n🫁 Generating Chest X-Ray dataset ({n_per_class} images/class) …")
    splits = {
        "train": int(n_per_class * 0.80),
        "val":   int(n_per_class * 0.10),
        "test":  n_per_class - int(n_per_class * 0.90),
    }
    total = 0
    for split, n in splits.items():
        for cls, gen_fn in CHEST_GEN.items():
            out = Path(base) / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                path = str(out / f"{cls}_{i:04d}.png")
                tmp  = str(out / f"_tmp_{i}.png")
                gen_fn(tmp, size)
                img = Image.open(tmp)
                if i > 0:
                    img = _augment(img)
                img.save(path)
                os.remove(tmp)
                total += 1
            print(f"   {split}/{cls}: {n} images")
    print(f"✅ Chest X-Ray: {total} images → {base}/")
    return total


def generate_samples(base="data/sample_images", size=256):
    """12 demo images shown in the Streamlit app."""
    os.makedirs(base, exist_ok=True)
    generators = {
        "chest_xray": chest_normal,
        "brain_mri":  brain_notumor,
        "ct_scan":    chest_pneumonia,
    }
    for name, fn in generators.items():
        fn(f"{base}/{name}_sample.png", size)
        for i in range(1, 4):
            fn(f"{base}/{name}_{i:02d}.png", size)
    print(f"✅ Sample images → {base}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic medical training images")
    ap.add_argument("--type",    choices=["brain","chest","samples","all"], default="all")
    ap.add_argument("--samples", type=int, default=150, help="Images per class")
    ap.add_argument("--size",    type=int, default=224, help="Image size (px)")
    args = ap.parse_args()

    print("🏥 MedAI — Synthetic Dataset Generator")
    print("="*45)
    print(f"   Images/class : {args.samples}")
    print(f"   Image size   : {args.size}×{args.size}")
    print("="*45)

    if args.type in ("brain", "all"):
        generate_brain(args.samples, args.size)
    if args.type in ("chest", "all"):
        generate_chest(args.samples, args.size)
    if args.type in ("samples", "all"):
        generate_samples()

    print("\n✅ Done! Run training next:")
    print("   python train_brain.py")
    print("   python train_chest.py")
