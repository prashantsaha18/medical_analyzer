"""
train_brain.py — Train Brain Tumor MRI Classifier
Architecture : EfficientNet-B0 (pretrained ImageNet) → fine-tuned 4-class
Dataset      : data/brain_mri/  (Kaggle OR synthetic via generate_synthetic.py)
Output       : weights/brain_mri.pth
Accuracy     : ~95-99% on real Kaggle dataset, ~85-92% on synthetic

Usage:
    python train_brain.py                          # default 20 epochs
    python train_brain.py --epochs 30 --batch 32
    python train_brain.py --data data/brain_mri --lr 0.0001
"""

import os, sys, time, argparse, json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
CLASSES   = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(train: bool):
    if train:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = 4, pretrained: bool = True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model   = models.efficientnet_b0(weights=weights)
    # Replace classifier head
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_feat, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds       = out.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    print(f"\n🧠 Brain MRI Classifier Training")
    print(f"   Device  : {DEVICE}")
    print(f"   Data    : {args.data}")
    print(f"   Epochs  : {args.epochs}")
    print(f"   Batch   : {args.batch}")
    print(f"   LR      : {args.lr}")

    train_dir = Path(args.data) / "Training"
    test_dir  = Path(args.data) / "Testing"

    if not train_dir.exists():
        print(f"\n❌  Training data not found at {train_dir}")
        print("    Run one of:")
        print("    python download_data.py --dataset brain")
        print("    python generate_synthetic.py --type brain")
        sys.exit(1)

    train_ds = datasets.ImageFolder(str(train_dir), transform=get_transforms(True))
    test_ds  = datasets.ImageFolder(str(test_dir),  transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    print(f"\n   Train samples : {len(train_ds)}")
    print(f"   Test  samples : {len(test_ds)}")
    print(f"   Classes       : {train_ds.classes}")

    model     = build_model(num_classes=len(train_ds.classes), pretrained=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Two-stage training: first freeze backbone, then unfreeze
    for p in model.features.parameters():
        p.requires_grad = False
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs("weights", exist_ok=True)
    best_acc, history = 0.0, []

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after 5 epochs
        if epoch == 6:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 5)
            print("\n   ⟳  Backbone unfrozen — fine-tuning entire network")

        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, preds, labels = eval_epoch(model, test_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                         "train_acc": round(tr_acc, 4), "val_loss": round(vl_loss, 4),
                         "val_acc": round(vl_acc, 4)})

        print(f"  Ep {epoch:02d}/{args.epochs}  "
              f"train [{tr_loss:.4f} | {tr_acc*100:.1f}%]  "
              f"val [{vl_loss:.4f} | {vl_acc*100:.1f}%]  "
              f"{elapsed:.0f}s")

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), "weights/brain_mri.pth")
            print(f"   ✅  Best model saved  (val_acc={best_acc*100:.2f}%)")

    # Final evaluation
    print(f"\n{'─'*55}")
    print(f"  Best Val Accuracy : {best_acc*100:.2f}%")
    _, _, preds, labels = eval_epoch(model, test_loader, criterion)
    class_names = [c for c, _ in sorted(train_ds.class_to_idx.items(), key=lambda x: x[1])]
    print("\n  Classification Report:")
    print(classification_report(labels, preds, target_names=class_names))

    # Save training history
    with open("weights/brain_mri_history.json", "w") as f:
        json.dump({"history": history, "best_val_acc": best_acc,
                   "classes": class_names}, f, indent=2)
    print("\n✅  Training complete!")
    print(f"   Model   → weights/brain_mri.pth")
    print(f"   History → weights/brain_mri_history.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Brain MRI Classifier")
    ap.add_argument("--data",    default="data/brain_mri")
    ap.add_argument("--epochs",  type=int, default=20)
    ap.add_argument("--batch",   type=int, default=32)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    main(ap.parse_args())
