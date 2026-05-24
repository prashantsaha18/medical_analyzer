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
    note      = " [Attending AI — clinical correlation required]" if is_demo else ""

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

    css_template = """<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
* {box-sizing:border-box}

/* Background and main container */
[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at 10% 20%, __BG__ 0%, rgba(15, 23, 42, 0.98) 90%) !important;
  font-family: 'Outfit', sans-serif;
  color: __TXT__;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
  background: __SBG__ !important;
  border-right: 1px solid __BRD__ !important;
}

[data-testid="stHeader"] {
  background: rgba(15, 23, 42, 0.4) !important;
  backdrop-filter: blur(8px);
  border-bottom: 1px solid __BRD__;
}

h1, h2, h3, h4 {
  color: __TXT__ !important;
  font-weight: 700;
  letter-spacing: -0.02em;
}

/* Premium cards */
.med-card {
  background: rgba(30, 41, 59, 0.4);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 16px;
  box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(10px);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.med-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
  border-color: rgba(255, 255, 255, 0.12);
}

/* Dashboard Metrics Box */
.mbox {
  background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
  border: 1px solid __BRD__;
  border-radius: 16px;
  padding: 20px;
  text-align: center;
  margin-bottom: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.mbox:hover {
  transform: scale(1.03) translateY(-3px);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
  border-color: #3b82f6;
}
.mbox-num {
  font-size: 2.2rem;
  font-weight: 800;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: -0.05em;
  margin-bottom: 4px;
}
.mbox-lbl {
  font-size: 0.75rem;
  font-weight: 600;
  color: __TXT2__;
  text-transform: uppercase;
  letter-spacing: 1.2px;
}

/* Modern Inputs and Selectboxes */
div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea, div[data-baseweb="select"] > div {
  background: rgba(15, 23, 42, 0.6) !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  border-radius: 12px !important;
  color: __TXT__ !important;
  font-size: 0.9rem !important;
  padding: 12px 16px !important;
  transition: all 0.25s ease !important;
}
div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus, div[data-baseweb="select"] > div:focus-within {
  border-color: #388bfd !important;
  box-shadow: 0 0 0 3px rgba(56, 139, 253, 0.25) !important;
  background: rgba(15, 23, 42, 0.8) !important;
}

/* Labels */
[data-testid="stTextInput"] label, [data-testid="stSelectbox"] label,
[data-testid="stRadio"] label, [data-testid="stTextArea"] label {
  color: __TXT2__ !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.3px;
}

/* Custom Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(15, 23, 42, 0.5);
  border-radius: 12px;
  padding: 6px;
  gap: 6px;
  border: 1px solid __BRD__;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px !important;
  color: __TXT2__ !important;
  font-size: 0.9rem !important;
  font-weight: 600 !important;
  padding: 8px 16px !important;
  border: none !important;
  transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
  color: white !important;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
}

/* Buttons Styling */
div[data-testid="stButton"]>button, div[data-testid="stFormSubmitButton"]>button {
  border-radius: 12px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  padding: 10px 24px !important;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div[data-testid="stButton"]>button:hover, div[data-testid="stFormSubmitButton"]>button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2) !important;
}
div[data-testid="stButton"]>button[kind="primary"], div[data-testid="stFormSubmitButton"]>button[kind="primary"] {
  background: linear-gradient(135deg, #2563eb, #8957e5) !important;
  border: none !important;
  color: white !important;
  box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
}
div[data-testid="stButton"]>button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"]>button[kind="primary"]:hover {
  background: linear-gradient(135deg, #3b82f6, #9333ea) !important;
  box-shadow: 0 8px 22px rgba(37, 99, 235, 0.6) !important;
}

/* Premium Forms */
div[data-testid="stForm"] {
  background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.4) 100%) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 20px !important;
  padding: 30px !important;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3) !important;
  backdrop-filter: blur(16px) !important;
}

/* Table Styling */
.med-table, .scan-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.85rem;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid __BRD__;
  margin-top: 10px;
}
.med-table th, .scan-table th {
  background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
  color: __TXT__;
  padding: 14px 16px;
  text-align: left;
  font-weight: 600;
  border-bottom: 1px solid __BRD__;
}
.med-table td, .scan-table td {
  padding: 12px 16px;
  border-bottom: 1px solid __BRD__;
  background: rgba(30, 41, 59, 0.2);
  color: __TXT__;
}
.med-table tr:last-child td, .scan-table tr:last-child td {
  border-bottom: none;
}
.med-table tr:hover td, .scan-table tr:hover td {
  background: rgba(56, 139, 253, 0.1) !important;
}

/* Animations */
@keyframes pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.08); }
  100% { transform: scale(1); }
}
@keyframes float {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-8px); }
  100% { transform: translateY(0px); }
}
.pulse-icon {
  display: inline-block;
  animation: pulse 3s infinite ease-in-out;
}
.float-card {
  animation: float 5s infinite ease-in-out;
}

/* Streamlit elements overrides */
div[data-testid="stExpander"] {
  background: rgba(30, 41, 59, 0.4) !important;
  border: 1px solid __BRD__ !important;
  border-radius: 12px !important;
}
[data-testid="stFileUploader"] {
  background: rgba(30, 41, 59, 0.2) !important;
  border: 2px dashed rgba(56, 139, 253, 0.3) !important;
  border-radius: 16px !important;
  padding: 20px !important;
  transition: all 0.25s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: #388bfd !important;
  background: rgba(56, 139, 253, 0.05) !important;
}
</style>"""

    return (css_template
            .replace("__BG__", bg)
            .replace("__BG2__", bg2)
            .replace("__BG3__", bg3)
            .replace("__TXT__", txt)
            .replace("__TXT2__", txt2)
            .replace("__BRD__", brd)
            .replace("__SBG__", sbg)
            .replace("__IBG__", ibg)
            .replace("__IBRD__", ibrd))

pil_to_bytes = image_to_bytes
auto_detect_scan_type = auto_scan_type
SEV_STYLES = SEV_STYLE
