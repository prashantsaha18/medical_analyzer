"""
model.py — MedAI Inference Engine
Models  : ChestNet (DenseNet121) | BrainNet (EfficientNet-B0) | CTNet (ResNet50)
Extras  : Grad-CAM heatmap | CLAHE pre-processing | Demo mode (no weights needed)
"""
import os, hashlib
import numpy as np
import cv2
from PIL import Image

# ── Graceful torch import ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# LABELS
# ─────────────────────────────────────────────────────────────────────────────
CHEST_CLASSES = ["Normal", "Pneumonia"]           # chest_xray.pth
BRAIN_CLASSES = ["Glioma", "Meningioma",          # brain_mri.pth
                 "No Tumor", "Pituitary Tumor"]
CT_CLASSES    = ["Normal", "Hemorrhage",           # demo only
                 "Infarction", "Mass Lesion", "Edema"]

# Human-readable class → scan-type map
SCAN_LABELS = {
    "Chest X-Ray": CHEST_CLASSES,
    "Brain MRI":   BRAIN_CLASSES,
    "CT Scan":     CT_CLASSES,
}

SEVERITY_RULES = {
    # (scan_type, class_name) → severity  (else computed from probability)
    ("Chest X-Ray", "Pneumonia"):       "High",
    ("Chest X-Ray", "Normal"):          "Normal",
    ("Brain MRI",   "No Tumor"):        "Normal",
    ("Brain MRI",   "Glioma"):          "Critical",
    ("Brain MRI",   "Meningioma"):      "High",
    ("Brain MRI",   "Pituitary Tumor"): "Moderate",
    ("CT Scan",     "Normal"):          "Normal",
    ("CT Scan",     "Hemorrhage"):      "Critical",
    ("CT Scan",     "Infarction"):      "Critical",
    ("CT Scan",     "Mass Lesion"):     "High",
    ("CT Scan",     "Edema"):           "Moderate",
}


def _severity(scan_type, top_class, prob):
    key = (scan_type, top_class)
    if key in SEVERITY_RULES:
        return SEVERITY_RULES[key]
    if prob > 0.80: return "Critical"
    if prob > 0.60: return "High"
    if prob > 0.40: return "Moderate"
    return "Low"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES  (only defined when torch is present)
# ─────────────────────────────────────────────────────────────────────────────
if TORCH_OK:
    class ChestNet(nn.Module):
        """DenseNet121 — 2-class (Normal / Pneumonia)."""
        def __init__(self, num_classes=2):
            super().__init__()
            self.net = models.densenet121(weights=None)
            nf = self.net.classifier.in_features
            self.net.classifier = nn.Sequential(
                nn.Dropout(0.4), nn.Linear(nf, 256),
                nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

        def forward(self, x):
            return self.net(x)

        def cam_layer(self):
            return self.net.features.denseblock4

    class BrainNet(nn.Module):
        """EfficientNet-B0 — 4-class brain tumor."""
        def __init__(self, num_classes=4):
            super().__init__()
            self.net = models.efficientnet_b0(weights=None)
            nf = self.net.classifier[1].in_features
            self.net.classifier = nn.Sequential(
                nn.Dropout(0.4), nn.Linear(nf, 256),
                nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

        def forward(self, x):
            return self.net(x)

        def cam_layer(self):
            return self.net.features[-1]

    class CTNet(nn.Module):
        """ResNet50 — 5-class CT scan."""
        def __init__(self, num_classes=5):
            super().__init__()
            self.net = models.resnet50(weights=None)
            nf = self.net.fc.in_features
            self.net.fc = nn.Sequential(
                nn.Dropout(0.4), nn.Linear(nf, 256),
                nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

        def forward(self, x):
            return self.net(x)

        def cam_layer(self):
            return self.net.layer4

    # Transform
    _TF = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def _to_tensor(pil_img):
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return _TF(pil_img).unsqueeze(0)

    # ── Grad-CAM ─────────────────────────────────────────────────────────────
    class GradCAM:
        def __init__(self, model, layer):
            self._acts = self._grads = None
            self._h = [
                layer.register_forward_hook(
                    lambda m, i, o: setattr(self, '_acts', o.detach())),
                layer.register_full_backward_hook(
                    lambda m, gi, go: setattr(self, '_grads', go[0].detach())),
            ]

        def compute(self, tensor, class_idx=None):
            tensor = tensor.requires_grad_(True)
            out    = self._model_fwd(tensor)
            if class_idx is None:
                class_idx = out.argmax(1).item()
            self._model_fwd.__self__.zero_grad() if hasattr(self._model_fwd, '__self__') else None
            out[0, class_idx].backward()
            if self._grads is None or self._acts is None:
                return None
            weights = self._grads[0].mean(dim=(1, 2))
            cam     = (weights[:, None, None] * self._acts[0]).sum(0)
            cam     = torch.relu(cam).detach().numpy()
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            return cam

        def remove(self):
            for h in self._h:
                h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CACHE + LOADER
# ─────────────────────────────────────────────────────────────────────────────
_CACHE: dict = {}

_MODEL_MAP = {
    "Chest X-Ray": ("weights/chest_xray.pth", "ChestNet", len(CHEST_CLASSES)),
    "Brain MRI":   ("weights/brain_mri.pth",  "BrainNet", len(BRAIN_CLASSES)),
    "CT Scan":     ("weights/ct_scan.pth",     "CTNet",    len(CT_CLASSES)),
} if TORCH_OK else {}

_CLASS_MAP = {
    "ChestNet": ChestNet if TORCH_OK else None,
    "BrainNet": BrainNet if TORCH_OK else None,
    "CTNet":    CTNet    if TORCH_OK else None,
} if TORCH_OK else {}


def _load(scan_type: str):
    if not TORCH_OK or scan_type not in _MODEL_MAP:
        return None
    if scan_type in _CACHE:
        return _CACHE[scan_type]
    path, cls_name, n = _MODEL_MAP[scan_type]
    if not os.path.exists(path):
        return None
    try:
        cls   = _CLASS_MAP[cls_name]
        model = cls(n)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()
        _CACHE[scan_type] = model
        print(f"✅ Loaded {scan_type} weights from {path}")
        return model
    except Exception as e:
        print(f"⚠️  Could not load {path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DEMO PREDICTOR  (deterministic, image-hash seeded)
# ─────────────────────────────────────────────────────────────────────────────
def _img_seed(pil_img: Image.Image) -> int:
    arr = np.array(pil_img.convert("L").resize((32, 32)))
    return int(hashlib.md5(arr.tobytes()).hexdigest()[:8], 16) % (2**31)


def _demo_predict(pil_img: Image.Image, scan_type: str) -> dict:
    labels = SCAN_LABELS[scan_type]
    rng    = np.random.default_rng(_img_seed(pil_img))

    if scan_type == "Brain MRI":
        # Realistic: most images are "No Tumor" in real data
        alpha = [0.5, 0.3, 2.0, 0.4]
    elif scan_type == "Chest X-Ray":
        alpha = [1.5, 0.6]   # bias toward Normal
    else:
        alpha = [1.8] + [0.4] * (len(labels) - 1)

    raw   = rng.dirichlet(alpha)
    probs = raw.tolist()
    top   = sorted(zip(labels, probs), key=lambda x: -x[1])
    sev   = _severity(scan_type, top[0][0], top[0][1])

    return {
        "type":        scan_type,
        "predictions": dict(zip(labels, [round(p, 4) for p in probs])),
        "top":         [(c, round(p, 4)) for c, p in top],
        "severity":    sev,
        "model":       f"Demo Mode — upload weights/{scan_type.lower().replace(' ','_')}.pth",
        "demo":        True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REAL PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
def _real_predict(pil_img: Image.Image, scan_type: str, model) -> dict:
    labels = SCAN_LABELS[scan_type]
    tensor = _to_tensor(pil_img)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].numpy()
    top  = sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])
    sev  = _severity(scan_type, top[0][0], top[0][1])
    return {
        "type":        scan_type,
        "predictions": dict(zip(labels, [round(float(p), 4) for p in probs])),
        "top":         [(c, round(float(p), 4)) for c, p in top],
        "severity":    sev,
        "model":       f"{type(model).__name__} [Trained Weights]",
        "demo":        False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def predict(pil_img: Image.Image, scan_type: str) -> dict:
    model = _load(scan_type)
    if model is not None:
        return _real_predict(pil_img, scan_type, model)
    return _demo_predict(pil_img, scan_type)


def generate_gradcam(pil_img: Image.Image, scan_type: str,
                     result: dict) -> np.ndarray | None:
    """Returns BGR heatmap overlaid on original, or None."""
    if not TORCH_OK:
        return None
    try:
        model = _load(scan_type)
        if model is None:
            # Use untrained ResNet for structural visual explanations (no download needed)
            backbone = models.resnet50(weights=None)
            backbone.eval()
            target_layer = backbone.layer4
        else:
            backbone     = model
            target_layer = model.cam_layer()

        tensor   = _to_tensor(pil_img)
        acts_box = [None]
        grad_box = [None]

        def fwd_hook(m, i, o):  acts_box[0] = o.detach()
        def bwd_hook(m, gi, go): grad_box[0] = go[0].detach()

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        tensor = tensor.requires_grad_(True)
        out    = backbone(tensor)
        backbone.zero_grad()

        top_labels = result.get("top", [])
        if top_labels and not result.get("demo", True) and model is not None:
            labels     = SCAN_LABELS[scan_type]
            top_name   = top_labels[0][0]
            cidx       = labels.index(top_name) if top_name in labels else out.argmax(1).item()
        else:
            cidx = out.argmax(1).item()

        out[0, cidx].backward()
        h1.remove(); h2.remove()

        acts = acts_box[0]; grads = grad_box[0]
        if acts is None or grads is None:
            return None

        w   = grads[0].mean(dim=(1, 2))
        cam = (w[:, None, None] * acts[0]).sum(0)
        cam = torch.relu(cam).detach().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        orig_np = np.array(pil_img.convert("RGB"))
        h, w_px = orig_np.shape[:2]
        hm      = cv2.resize(cam, (w_px, h))
        hm      = np.uint8(255 * hm)
        hm_col  = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        hm_rgb  = cv2.cvtColor(hm_col, cv2.COLOR_BGR2RGB)
        overlay = (orig_np * 0.55 + hm_rgb * 0.45).astype(np.uint8)
        return overlay

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


def weights_status() -> dict:
    """Returns dict of which model weights are loaded."""
    status = {}
    for scan_type, (path, _, _) in _MODEL_MAP.items():
        status[scan_type] = os.path.exists(path)
    return status
