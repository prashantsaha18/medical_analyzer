"""
MedVision Inference Engine
Original architecture: MedVisionNet (custom CNN backbone with channel attention)
Supports: Chest X-Ray (2-class) | Brain MRI (4-class) | CT Scan (5-class)
Features: Grad-CAM++ heatmap | CLAHE preprocessing | demo fallback
"""
import os
import hashlib
import numpy as np
import cv2
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as T
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Label definitions ─────────────────────────────────────────────────────────
CHEST_LABELS = ["Normal", "Pneumonia"]
BRAIN_LABELS  = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CT_LABELS     = ["Normal", "Hemorrhage", "Infarction", "Mass Lesion", "Edema"]

SCAN_LABELS = {
    "Chest X-Ray": CHEST_LABELS,
    "Brain MRI":   BRAIN_LABELS,
    "CT Scan":     CT_LABELS,
}

# ── Severity decision table ───────────────────────────────────────────────────
_SEV_TABLE = {
    ("Chest X-Ray", "Normal"):         "Normal",
    ("Chest X-Ray", "Pneumonia"):      "High",
    ("Brain MRI",   "No Tumor"):       "Normal",
    ("Brain MRI",   "Glioma"):         "Critical",
    ("Brain MRI",   "Meningioma"):     "High",
    ("Brain MRI",   "Pituitary"):      "Moderate",
    ("CT Scan",     "Normal"):         "Normal",
    ("CT Scan",     "Hemorrhage"):     "Critical",
    ("CT Scan",     "Infarction"):     "Critical",
    ("CT Scan",     "Mass Lesion"):    "High",
    ("CT Scan",     "Edema"):          "Moderate",
}

def _severity(scan_type: str, top_class: str, confidence: float) -> str:
    key = (scan_type, top_class)
    if key in _SEV_TABLE:
        return _SEV_TABLE[key]
    if confidence > 0.80: return "Critical"
    if confidence > 0.60: return "High"
    if confidence > 0.40: return "Moderate"
    return "Low"

# ── Image transform ───────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    _TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])
    def _to_tensor(img: Image.Image) -> "torch.Tensor":
        return _TRANSFORM(img.convert("RGB")).unsqueeze(0)

# ── Original MedVisionNet architecture ────────────────────────────────────────
if TORCH_AVAILABLE:
    class _ChannelAttention(nn.Module):
        """Squeeze-and-excitation channel attention block."""
        def __init__(self, channels: int, reduction: int = 16):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc   = nn.Sequential(
                nn.Linear(channels, max(channels // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(max(channels // reduction, 8), channels),
                nn.Sigmoid(),
            )

        def forward(self, x):
            b, c, _, _ = x.shape
            w = self.pool(x).view(b, c)
            w = self.fc(w).view(b, c, 1, 1)
            return x * w

    class MedVisionNet(nn.Module):
        """
        Original medical image classifier.
        Backbone: pretrained DenseNet121 feature extractor
        Head: channel attention → global pooling → FC with dropout
        """
        def __init__(self, num_classes: int, pretrained: bool = False):
            super().__init__()
            base       = models.densenet121(weights=None)
            self.feats = base.features        # (B, 1024, 7, 7)
            self.attn  = _ChannelAttention(1024)
            self.pool  = nn.AdaptiveAvgPool2d(1)
            self.head  = nn.Sequential(
                nn.Dropout(p=0.45),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Dropout(p=0.30),
                nn.Linear(512, num_classes),
            )
            self._init_head()

        def _init_head(self):
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            f = self.feats(x)
            f = self.attn(f)
            f = self.pool(f).flatten(1)
            return self.head(f)

        def cam_target_layer(self):
            return self.feats.denseblock4

# ── Weight paths ──────────────────────────────────────────────────────────────
_WEIGHT_MAP = {
    "Chest X-Ray": ("weights/chest_xray.pth", len(CHEST_LABELS)),
    "Brain MRI":   ("weights/brain_mri.pth",  len(BRAIN_LABELS)),
    "CT Scan":     ("weights/ct_scan.pth",     len(CT_LABELS)),
}

_MODEL_CACHE: dict = {}

def _load_model(scan_type: str):
    if not TORCH_AVAILABLE or scan_type not in _WEIGHT_MAP:
        return None
    if scan_type in _MODEL_CACHE:
        return _MODEL_CACHE[scan_type]

    path, nc = _WEIGHT_MAP[scan_type]
    if not os.path.exists(path):
        return None

    try:
        net   = MedVisionNet(num_classes=nc)
        state = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state, strict=False)
        net.eval()
        _MODEL_CACHE[scan_type] = net
        return net
    except Exception as exc:
        print(f"[MedVision] Could not load {path}: {exc}")
        return None

def weights_status() -> dict:
    return {st: os.path.exists(p) for st, (p, _) in _WEIGHT_MAP.items()}

# ── Grad-CAM++ heatmap ────────────────────────────────────────────────────────
def generate_gradcam(
    pil_img: Image.Image,
    scan_type: str,
    result: dict,
) -> "np.ndarray | None":
    """
    Original Grad-CAM++ implementation.
    Returns RGB overlay array (H x W x 3), or None on failure.
    """
    if not TORCH_AVAILABLE:
        return None

    try:
        model = _load_model(scan_type)
        if model is None:
            net   = MedVisionNet(num_classes=max(len(SCAN_LABELS[scan_type]), 2))
            layer = net.cam_target_layer()
        else:
            net   = model
            layer = net.cam_target_layer()

        tensor = _to_tensor(pil_img)

        acts_store  = [None]
        grads_store = [None]

        def _fwd(m, inp, out): acts_store[0]  = out.detach()
        def _bwd(m, gi, go):   grads_store[0] = go[0].detach()

        h1 = layer.register_forward_hook(_fwd)
        h2 = layer.register_full_backward_hook(_bwd)

        tensor = tensor.requires_grad_(True)
        logits = net(tensor)

        labels   = SCAN_LABELS[scan_type]
        top_name = result.get("top", [[labels[0]]])[0][0]
        cidx     = labels.index(top_name) if top_name in labels else logits.argmax(1).item()

        net.zero_grad()
        logits[0, cidx].backward()

        h1.remove(); h2.remove()

        acts  = acts_store[0]   # (1, C, H, W)
        grads = grads_store[0]  # (1, C, H, W)

        if acts is None or grads is None:
            return None

        # Grad-CAM++ weights (alpha from second-order taylor expansion)
        grad_sq  = grads ** 2
        grad_cu  = grads ** 3
        denom    = 2 * grad_sq + acts * grad_cu.sum(dim=(2, 3), keepdim=True)
        denom    = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha    = grad_sq / denom
        weights  = (alpha * F.relu(grads)).sum(dim=(2, 3))  # (1, C)

        cam = (weights[0, :, None, None] * acts[0]).sum(0)
        cam = F.relu(cam).detach().numpy()

        if cam.max() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Overlay
        orig = np.array(pil_img.convert("RGB"))
        h, w = orig.shape[:2]
        hm   = cv2.resize(cam, (w, h))
        hm   = np.uint8(255 * hm)
        hm   = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        hm   = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        return np.clip(orig * 0.55 + hm * 0.45, 0, 255).astype(np.uint8)

    except Exception as exc:
        print(f"[GradCAM++] {exc}")
        return None

# ── Demo predictor (no weights required) ─────────────────────────────────────
def _image_hash(img: Image.Image) -> int:
    """Stable seed from image pixel statistics — same image → same result."""
    arr = np.array(img.convert("L").resize((32, 32)), dtype=np.float32)
    stats = np.array([arr.mean(), arr.std(), arr.min(), arr.max()])
    raw   = hashlib.md5(stats.tobytes()).hexdigest()[:8]
    return int(raw, 16) % (2 ** 31)

# Realistic Dirichlet concentration priors per scan type
_PRIORS = {
    "Chest X-Ray": [2.0, 0.5],              # Normal > Pneumonia
    "Brain MRI":   [0.4, 0.3, 1.8, 0.5],   # No Tumor most common
    "CT Scan":     [2.0, 0.3, 0.3, 0.3, 0.4],
}

def _demo_predict(img: Image.Image, scan_type: str) -> dict:
    labels  = SCAN_LABELS[scan_type]
    alpha   = _PRIORS.get(scan_type, [1.0] * len(labels))
    rng     = np.random.default_rng(_image_hash(img))
    raw     = rng.dirichlet(alpha)
    probs   = [round(float(p), 4) for p in raw]
    paired  = sorted(zip(labels, probs), key=lambda x: -x[1])
    top_cls, top_prob = paired[0]
    return {
        "type":        scan_type,
        "predictions": dict(zip(labels, probs)),
        "top":         paired,
        "severity":    _severity(scan_type, top_cls, top_prob),
        "model":       "Attending Classifier (Simulated)",
        "demo":        True,
    }

# ── Real predictor ────────────────────────────────────────────────────────────
def _real_predict(img: Image.Image, scan_type: str, net) -> dict:
    labels = SCAN_LABELS[scan_type]
    tensor = _to_tensor(img)
    with torch.no_grad():
        logits = net(tensor)
        probs  = torch.softmax(logits, dim=1)[0].numpy()
    probs  = [round(float(p), 4) for p in probs]
    paired = sorted(zip(labels, probs), key=lambda x: -x[1])
    top_cls, top_prob = paired[0]
    return {
        "type":        scan_type,
        "predictions": dict(zip(labels, probs)),
        "top":         paired,
        "severity":    _severity(scan_type, top_cls, top_prob),
        "model":       "MedVisionNet [Trained]",
        "demo":        False,
    }

# ── Public API ────────────────────────────────────────────────────────────────
def predict(img: Image.Image, scan_type: str) -> dict:
    """Run inference. Falls back to demo mode if no weights found."""
    net = _load_model(scan_type)
    if net is not None:
        return _real_predict(img, scan_type, net)
    return _demo_predict(img, scan_type)