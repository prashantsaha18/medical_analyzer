"""
train_chest.py — Train Chest X-Ray Classifier
Architecture : DenseNet121 (pretrained ImageNet) → fine-tuned 2-class
Dataset      : data/chest_xray/  (Kaggle paultimothymooney/chest-xray-pneumonia ~1.2GB)
               OR synthetic via: python generate_synthetic.py --type chest
Output       : weights/chest_xray.pth

Usage:
    python train_chest.py
    python train_chest.py --epochs 15 --batch 32
    python train_chest.py --data data/chest_xray --lr 0.0001
"""

import os, sys, time, argparse, json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["NORMAL", "PNEUMONIA"]


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS  (chest X-ray specific augmentations)
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(train: bool):
    if train:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.RandomAffine(degrees=0, shear=5),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  (DenseNet121 — same family as CheXNet)
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = 2, pretrained: bool = True):
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model   = models.densenet121(weights=weights)
    in_feat = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_feat, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTED SAMPLER  (handles class imbalance — pneumonia > normal)
# ─────────────────────────────────────────────────────────────────────────────
def make_weighted_sampler(dataset):
    counts  = np.bincount([label for _, label in dataset.samples])
    weights = 1.0 / counts
    sample_weights = torch.tensor(
        [weights[label] for _, label in dataset.samples], dtype=torch.float)
    return torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOOPS
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
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_probs, all_labels = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out          = model(imgs)
        loss         = criterion(out, labels)
        total_loss  += loss.item() * imgs.size(0)
        probs        = torch.softmax(out, dim=1)[:, 1]
        preds        = out.argmax(1)
        correct     += (preds == labels).sum().item()
        total       += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_probs, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    print(f"\n🫁 Chest X-Ray Classifier Training")
    print(f"   Device  : {DEVICE}")
    print(f"   Data    : {args.data}")
    print(f"   Epochs  : {args.epochs}")

    train_dir = Path(args.data) / "train"
    val_dir   = Path(args.data) / "val"
    test_dir  = Path(args.data) / "test"

    if not train_dir.exists():
        print(f"\n❌  Training data not found at {train_dir}")
        print("    Run one of:")
        print("    python download_data.py --dataset chest")
        print("    python generate_synthetic.py --type chest")
        sys.exit(1)

    train_ds = datasets.ImageFolder(str(train_dir), transform=get_transforms(True))
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=get_transforms(False)) \
               if val_dir.exists() else None
    test_ds  = datasets.ImageFolder(str(test_dir),  transform=get_transforms(False))

    sampler      = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              sampler=sampler, num_workers=args.workers)
    val_loader   = DataLoader(val_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=args.workers) if val_ds else None
    test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers)

    print(f"\n   Train : {len(train_ds)} | Test : {len(test_ds)}")
    print(f"   Classes : {train_ds.classes}")

    model     = build_model(num_classes=len(train_ds.classes), pretrained=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, "max", patience=3, factor=0.5, verbose=True)

    os.makedirs("weights", exist_ok=True)
    best_acc, history = 0.0, []

    for epoch in range(1, args.epochs + 1):
        # Unfreeze full backbone after epoch 4
        if epoch == 5:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            print("\n   ⟳  Full model unfrozen")

        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)

        eval_loader = val_loader if val_loader else test_loader
        vl_loss, vl_acc, preds, probs, labels = eval_epoch(
            model, eval_loader, criterion)

        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.0

        scheduler.step(vl_acc)
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                         "train_acc": round(tr_acc, 4), "val_loss": round(vl_loss, 4),
                         "val_acc": round(vl_acc, 4), "auc": round(auc, 4)})

        print(f"  Ep {epoch:02d}/{args.epochs}  "
              f"train [{tr_loss:.4f}|{tr_acc*100:.1f}%]  "
              f"val [{vl_loss:.4f}|{vl_acc*100:.1f}%]  "
              f"AUC {auc:.3f}  {elapsed:.0f}s")

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), "weights/chest_xray.pth")
            print(f"   ✅  Best model saved (val_acc={best_acc*100:.2f}%)")

    # Final test eval
    print(f"\n{'─'*55}")
    _, te_acc, preds, probs, labels = eval_epoch(model, test_loader, criterion)
    print(f"  Test Accuracy : {te_acc*100:.2f}%")
    class_names = [c for c, _ in sorted(train_ds.class_to_idx.items(), key=lambda x: x[1])]
    print(classification_report(labels, preds, target_names=class_names))

    with open("weights/chest_xray_history.json", "w") as f:
        json.dump({"history": history, "best_val_acc": best_acc,
                   "test_acc": te_acc, "classes": class_names}, f, indent=2)

    print("\n✅  Training complete!")
    print(f"   Model   → weights/chest_xray.pth")
    print(f"   History → weights/chest_xray_history.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Chest X-Ray Classifier")
    ap.add_argument("--data",    default="data/chest_xray")
    ap.add_argument("--epochs",  type=int,   default=10)
    ap.add_argument("--batch",   type=int,   default=32)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--workers", type=int,   default=2)
    main(ap.parse_args())
