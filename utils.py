"""utils.py — Image processing, CSS, UI helpers for MedAI"""
import streamlit as st
import numpy as np, cv2, io, base64, random, string
from PIL import Image
from datetime import datetime, date

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE
# ─────────────────────────────────────────────────────────────────────────────
def load_image(src) -> Image.Image:
    return Image.open(src).convert("RGB")

def clahe_enhance(pil_img: Image.Image) -> Image.Image:
    gray    = np.array(pil_img.convert("L"))
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh     = clahe.apply(gray)
    return Image.fromarray(cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB))

def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))

def image_to_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO(); pil_img.save(buf, "PNG"); return buf.getvalue()

def auto_scan_type(filename: str) -> str:
    n = filename.lower()
    if any(k in n for k in ["chest","xray","x-ray","cxr","thorax","lung"]): return "Chest X-Ray"
    if any(k in n for k in ["brain","mri","tumor","glioma","neuro"]):        return "Brain MRI"
    if any(k in n for k in ["ct","scan","abdomen","spine","head"]):           return "CT Scan"
    return "Chest X-Ray"

def generate_patient_id() -> str:
    ts   = datetime.now().strftime("%y%m%d")
    rand = "".join(random.choices(string.digits, k=4))
    return f"PAT-{ts}-{rand}"

# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY
# ─────────────────────────────────────────────────────────────────────────────
SEV_STYLE = {
    "Critical": {"bg":"#fef2f2","border":"#ef4444","text":"#b91c1c","icon":"🔴","color":"#ef4444"},
    "High":     {"bg":"#fff7ed","border":"#f97316","text":"#c2410c","icon":"🟠","color":"#f97316"},
    "Moderate": {"bg":"#fefce8","border":"#eab308","text":"#a16207","icon":"🟡","color":"#eab308"},
    "Low":      {"bg":"#f0fdf4","border":"#22c55e","text":"#15803d","icon":"🟢","color":"#22c55e"},
    "Normal":   {"bg":"#eff6ff","border":"#3b82f6","text":"#1d4ed8","icon":"🔵","color":"#3b82f6"},
}

def sev_badge(severity: str, dark=True) -> str:
    c = SEV_STYLE.get(severity, SEV_STYLE["Low"])
    bg = c["color"]
    return (f'<span style="display:inline-block;padding:3px 12px;border-radius:999px;'
            f'background:{bg};color:white;font-size:.72rem;font-weight:700">'
            f'{c["icon"]} {severity}</span>')

def confidence_bar(label: str, prob: float, color="#3b82f6") -> str:
    pct = min(int(prob*100), 100)
    return (f'<div style="margin-bottom:8px">'
            f'<div style="display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:3px">'
            f'<span style="font-weight:600">{label}</span>'
            f'<span style="color:{color};font-weight:700">{pct}%</span></div>'
            f'<div style="background:#334155;border-radius:4px;height:8px">'
            f'<div style="background:{color};width:{pct}%;height:8px;border-radius:4px"></div>'
            f'</div></div>')

# ─────────────────────────────────────────────────────────────────────────────
# FINDINGS TEXT
# ─────────────────────────────────────────────────────────────────────────────
def build_findings(result: dict) -> str:
    scan_type = result.get("type","")
    top       = result.get("top", [])
    is_demo   = result.get("demo", True)
    note      = " [AI Demo — load trained weights for clinical use]" if is_demo else ""

    if not top:
        return "Insufficient data for analysis."

    top_cls, top_prob = top[0]

    if scan_type == "Chest X-Ray":
        if top_cls == "Normal":
            return (f"Lung fields appear clear bilaterally. No significant consolidation, "
                    f"pleural effusion, or pneumothorax identified. Cardiac silhouette within "
                    f"normal limits. Bony thorax intact.{note}")
        return (f"Chest X-Ray Analysis:\n• {top_cls} identified with {top_prob*100:.1f}% confidence.\n"
                f"• Lung fields show evidence of opacification. Clinical correlation recommended.\n"
                f"• Comparison with prior imaging advised.{note}")

    if scan_type == "Brain MRI":
        if top_cls == "No Tumor":
            return (f"No intracranial mass lesion identified. Brain parenchyma appears normal. "
                    f"Ventricles and sulci of normal calibre. No midline shift detected.{note}")
        diff = ", ".join(f"{c} ({p*100:.0f}%)" for c,p in top[1:3])
        return (f"Brain MRI Analysis:\n• Primary finding: {top_cls} ({top_prob*100:.1f}% confidence).\n"
                f"• Differential: {diff}.\n• MRI with contrast and neurosurgical review recommended.{note}")

    # CT
    if top_cls == "Normal":
        return (f"CT scan demonstrates no acute intracranial abnormality. No haemorrhage, "
                f"mass effect, or midline shift. Grey-white matter differentiation preserved.{note}")
    return (f"CT Analysis:\n• {top_cls} identified ({top_prob*100:.1f}% confidence).\n"
            f"• Urgent radiologist and clinical review recommended.{note}")

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
def get_css(dark=True) -> str:
    if dark:
        bg="#0f172a"; bg2="#1e293b"; bg3="#334155"; txt="#f1f5f9"
        txt2="#94a3b8"; brd="#334155"; sbg="#0f172a"; ibg="#1e293b"; ibrd="#475569"
    else:
        bg="#f8fafc"; bg2="#ffffff"; bg3="#e2e8f0"; txt="#0f172a"
        txt2="#475569"; brd="#e2e8f0"; sbg="#f1f5f9"; ibg="#ffffff"; ibrd="#cbd5e1"

    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
*{{box-sizing:border-box}}
[data-testid="stAppViewContainer"]{{background:{bg}!important;font-family:'Inter',sans-serif;color:{txt}}}
[data-testid="stSidebar"]{{background:{sbg}!important;border-right:1px solid {brd}!important}}
[data-testid="stHeader"]{{background:{bg}!important;border-bottom:1px solid {brd}}}
h1,h2,h3{{color:{txt}!important;font-weight:700}}
.med-card{{background:{bg2};border:1px solid {brd};border-radius:14px;padding:20px;
  margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.metric-box{{background:{bg2};border:1px solid {brd};border-radius:12px;
  padding:16px;text-align:center}}
.metric-num{{font-size:1.9rem;font-weight:800;font-family:'JetBrains Mono',monospace}}
.metric-lbl{{font-size:.7rem;color:{txt2};text-transform:uppercase;letter-spacing:.8px;margin-top:2px}}
.scan-table{{width:100%;border-collapse:collapse;font-size:.82rem}}
.scan-table th{{background:#2563eb;color:white;padding:10px 12px;text-align:left;font-weight:600}}
.scan-table td{{padding:9px 12px;border-bottom:1px solid {brd};color:{txt}}}
.scan-table tr:hover td{{background:{bg3}}}
.stTabs [data-baseweb="tab-list"]{{background:{bg2};border-radius:10px;padding:3px;
  gap:2px;border:1px solid {brd}}}
.stTabs [data-baseweb="tab"]{{border-radius:8px!important;color:{txt2}!important;
  font-size:.82rem!important;font-weight:500!important}}
.stTabs [aria-selected="true"]{{background:#2563eb!important;color:white!important}}
div[data-baseweb="select"]>div{{background:{ibg}!important;border-color:{ibrd}!important;color:{txt}!important}}
[data-testid="stTextInput"] label,[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label,[data-testid="stTextArea"] label{{color:{txt2}!important;font-size:.82rem!important}}
div[data-testid="stButton"]>button{{border-radius:9px!important;
  font-family:'Inter',sans-serif!important;font-weight:600!important;transition:all .18s!important}}
div[data-testid="stButton"]>button[kind="primary"]{{
  background:linear-gradient(135deg,#2563eb,#1d4ed8)!important;
  border:none!important;color:white!important;
  box-shadow:0 4px 12px rgba(37,99,235,.3)!important}}
div[data-testid="stExpander"]{{background:{bg2}!important;
  border:1px solid {brd}!important;border-radius:11px!important}}
.stMetric{{background:{bg2}!important;border-radius:11px!important;
  border:1px solid {brd}!important;padding:14px!important}}
.stMetric label{{color:{txt2}!important;font-size:.7rem!important}}
[data-testid="stMetricValue"]{{color:{txt}!important;font-family:'JetBrains Mono',monospace!important}}
::-webkit-scrollbar{{width:5px}}::-webkit-scrollbar-track{{background:{bg}}}
::-webkit-scrollbar-thumb{{background:{bg3};border-radius:3px}}
[data-testid="stFileUploader"]{{background:{bg2}!important;
  border:2px dashed {ibrd}!important;border-radius:12px!important}}
.mbox{{background:{bg2};border:1px solid {brd};border-radius:12px;
  padding:16px 20px;text-align:center;margin-bottom:8px}}
.mbox-num{{font-size:1.9rem;font-weight:800;font-family:'JetBrains Mono',monospace;color:{txt}}}
.mbox-lbl{{font-size:.72rem;color:{txt2};text-transform:uppercase;letter-spacing:.8px;margin-top:4px}}
.med-table{{width:100%;border-collapse:collapse;font-size:.82rem}}
.med-table th{{background:#2563eb;color:white;padding:10px 12px;text-align:left;font-weight:600}}
.med-table td{{padding:9px 12px;border-bottom:1px solid {brd};color:{txt}}}
.med-table tr:hover td{{background:{bg3}}}
</style>"""

pil_to_bytes = image_to_bytes
auto_detect_scan_type = auto_scan_type
SEV_STYLES = SEV_STYLE
