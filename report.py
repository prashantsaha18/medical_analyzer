"""report.py — Doctor-style PDF medical report (ReportLab)"""
from datetime import datetime
from io import BytesIO
import numpy as np
from PIL import Image as PILImage

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable,
                                     Image as RLImage)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

NAVY  = "#0f172a"; BLUE  = "#2563eb"; LBLUE = "#eff6ff"
SEV_COLORS = {"Critical":"#ef4444","High":"#f97316",
               "Moderate":"#eab308","Low":"#22c55e","Normal":"#3b82f6"}

def _recs(sev, scan_type):
    base = {
        "Critical":["Immediate clinical intervention required.",
                    "Urgent specialist referral and hospital admission.",
                    "Repeat imaging within 24 h post-treatment."],
        "High":    ["Prompt specialist consultation within 48 h.",
                    "Additional imaging studies recommended.",
                    "Follow-up in 1 week."],
        "Moderate":["Outpatient specialist review within 2 weeks.",
                    "Follow-up imaging in 4–6 weeks.",
                    "Monitor symptoms closely."],
        "Low":     ["Routine follow-up in 3 months.",
                    "Maintain current management plan.",
                    "Report any new or worsening symptoms."],
        "Normal":  ["No acute findings. Continue routine health maintenance.",
                    "Annual imaging as per clinical guidelines."],
    }
    return base.get(sev, base["Low"])

def _pil_to_rl(pil_img, w_cm, h_cm):
    buf = BytesIO()
    pil_img.save(buf, format="PNG"); buf.seek(0)
    return RLImage(buf, width=w_cm*cm, height=h_cm*cm)

def generate_pdf_report(patient, result, findings, doctor_name,
                         heatmap_img=None, original_img=None) -> bytes:
    if not REPORTLAB_OK:
        return (
            f"MEDICAL IMAGING REPORT\n{'='*50}\n"
            f"Patient : {patient.get('name','N/A')}\n"
            f"ID      : {patient.get('patient_id','N/A')}\n"
            f"Date    : {datetime.now():%d %B %Y %H:%M}\n"
            f"Doctor  : {doctor_name}\n\n"
            f"Scan    : {result.get('type','N/A')}\n"
            f"Severity: {result.get('severity','N/A')}\n"
            f"Model   : {result.get('model','N/A')}\n\n"
            f"FINDINGS:\n{findings}\n\nRECOMMENDATIONS:\n"
            + "\n".join(f"• {r}" for r in _recs(result.get("severity","Low"), result.get("type","")))
            + "\n\n⚠ AI-assisted — requires clinical validation.\n"
        ).encode()

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=1.5*cm, bottomMargin=2*cm)
    SS  = getSampleStyleSheet()
    story = []
    sev   = result.get("severity","Unknown")
    sev_c = colors.HexColor(SEV_COLORS.get(sev,"#64748b"))

    def _p(text, style="Normal", **kw):
        s = ParagraphStyle(f"s{id(text)}", parent=SS[style], **kw)
        return Paragraph(text, s)

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = Table([[
        _p(f'<font color="{BLUE}" size="18"><b>MedAI Diagnostics</b></font><br/>'
           f'<font color="#64748b" size="8">AI-Assisted Medical Imaging Report</font>',
           alignment=TA_LEFT),
        _p(f'<font color="#64748b" size="8">Date: <b>{datetime.now():%d %B %Y}</b><br/>'
           f'Ref: {datetime.now():%Y%m%d%H%M%S}</font>',
           alignment=TA_RIGHT),
    ]], colWidths=["65%","35%"])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor(LBLUE)),
        ("TOPPADDING",(0,0),(-1,-1),12),("BOTTOMPADDING",(0,0),(-1,-1),12),
        ("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14),
    ]))
    story += [hdr, Spacer(1,.35*cm)]

    # ── Patient / Scan info ────────────────────────────────────────────────────
    info = [
        ["PATIENT INFO","","SCAN DETAILS",""],
        ["Name:",  patient.get("name","N/A"),  "Type:",    result.get("type","N/A")],
        ["ID:",    patient.get("patient_id","N/A"), "Model:", result.get("model","N/A")[:32]],
        ["DOB:",   str(patient.get("dob","N/A")), "Severity:", sev],
        ["Gender:",patient.get("gender","N/A"), "Date:", datetime.now().strftime("%d/%m/%Y")],
    ]
    it = Table(info, colWidths=["16%","32%","16%","36%"])
    it.setStyle(TableStyle([
        ("SPAN",(0,0),(1,0)),("SPAN",(2,0),(3,0)),
        ("BACKGROUND",(0,0),(1,0),colors.HexColor(NAVY)),
        ("BACKGROUND",(2,0),(3,0),colors.HexColor(NAVY)),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),8.5),
        ("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),("FONTNAME",(2,1),(2,-1),"Helvetica-Bold"),
        ("FONTSIZE",(0,1),(-1,-1),8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f8fafc")]),
        ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#e2e8f0")),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),8),
    ]))
    story += [it, Spacer(1,.3*cm)]

    # ── Severity banner ────────────────────────────────────────────────────────
    icons = {"Critical":"🔴","High":"🟠","Moderate":"🟡","Low":"🟢","Normal":"🔵"}
    sb = Table([[_p(f'<font size="11" color="white"><b>{icons.get(sev,"⚪")} SEVERITY: {sev.upper()}</b></font>',
                    alignment=TA_CENTER, backColor=sev_c)]], colWidths=["100%"])
    sb.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),sev_c),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
    ]))
    story += [sb, Spacer(1,.3*cm)]

    # ── Images ────────────────────────────────────────────────────────────────
    img_cells, captions = [], []
    if original_img is not None:
        img_cells.append(_pil_to_rl(original_img.resize((200,200)), 6.2, 7.2))
        captions.append(_p('<font size="8" color="#64748b">Original Image</font>',
                           alignment=TA_CENTER))
    if heatmap_img is not None:
        heat_pil = PILImage.fromarray(heatmap_img.astype(np.uint8))
        img_cells.append(_pil_to_rl(heat_pil.resize((200,200)), 6.2, 7.2))
        captions.append(_p('<font size="8" color="#64748b">Grad-CAM Heatmap</font>',
                           alignment=TA_CENTER))
    if img_cells:
        w = 8.5*cm
        it2 = Table([img_cells, captions], colWidths=[w]*len(img_cells))
        it2.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),
                                  ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
        story += [it2, Spacer(1,.3*cm)]

    # ── Findings ──────────────────────────────────────────────────────────────
    story.append(_p(f'<font color="{NAVY}" size="10.5"><b>AI ANALYSIS FINDINGS</b></font>',
                    backColor=colors.HexColor("#f1f5f9"), spaceAfter=4))
    story.append(HRFlowable(width="100%",thickness=1,color=colors.HexColor(BLUE),spaceAfter=5))
    for line in findings.split("\n"):
        if line.strip():
            story.append(_p(f'<font size="9">{line}</font>', leftIndent=10, spaceAfter=3))
    story.append(Spacer(1,.25*cm))

    # ── Probability table ──────────────────────────────────────────────────────
    top = result.get("top",[])[:8]
    if top:
        story.append(_p(f'<font color="{NAVY}" size="10.5"><b>PROBABILITY SCORES</b></font>',
                        backColor=colors.HexColor("#f1f5f9"), spaceAfter=4))
        story.append(HRFlowable(width="100%",thickness=1,color=colors.HexColor(BLUE),spaceAfter=5))
        rows = [["Condition / Class","Confidence","Level"]]
        for cond, prob in top:
            lbl = "HIGH" if prob>.6 else "MED" if prob>.3 else "LOW"
            lc  = "#ef4444" if prob>.6 else "#f97316" if prob>.3 else "#22c55e"
            rows.append([
                _p(f'<font size="8">{cond}</font>'),
                _p(f'<font size="8"><b>{prob*100:.1f}%</b></font>', alignment=TA_CENTER),
                _p(f'<font size="8" color="{lc}"><b>{lbl}</b></font>', alignment=TA_CENTER),
            ])
        pt = Table(rows, colWidths=["60%","20%","20%"])
        pt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),8.5),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f8fafc")]),
            ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#e2e8f0")),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),8),("ALIGN",(1,0),(-1,-1),"CENTER"),
        ]))
        story += [pt, Spacer(1,.25*cm)]

    # ── Recommendations ────────────────────────────────────────────────────────
    story.append(_p(f'<font color="{NAVY}" size="10.5"><b>CLINICAL RECOMMENDATIONS</b></font>',
                    backColor=colors.HexColor("#f1f5f9"), spaceAfter=4))
    story.append(HRFlowable(width="100%",thickness=1,color=colors.HexColor(BLUE),spaceAfter=5))
    for rec in _recs(sev, result.get("type","")):
        story.append(_p(f'<font size="9">• {rec}</font>', leftIndent=10, spaceAfter=4))

    story.append(Spacer(1,.5*cm))
    story.append(HRFlowable(width="100%",thickness=.5,color=colors.HexColor("#94a3b8")))
    story.append(Spacer(1,.2*cm))
    story.append(_p('<font size="7.5" color="#64748b"><b>⚠ DISCLAIMER:</b> This report is '
                    'AI-generated and must be validated by a qualified medical professional. '
                    'Not for standalone diagnostic use.</font>', alignment=TA_CENTER))
    story.append(Spacer(1,.4*cm))

    sig = Table([[
        _p(f'<font size="8"><b>{doctor_name}</b><br/>Attending Physician<br/>'
           f'Date: {datetime.now():%d/%m/%Y}</font>', alignment=TA_LEFT),
        _p('<font size="8">_______________________<br/>Signature &amp; Stamp</font>',
           alignment=TA_RIGHT),
    ]], colWidths=["50%","50%"])
    story.append(sig)
    doc.build(story)
    return buf.getvalue()


# ── Public aliases expected by app.py ─────────────────────────────────────────
RL_OK = REPORTLAB_OK


def generate_pdf(patient, result, findings, doctor_name,
                 heatmap_arr=None, original_pil=None) -> bytes:
    """Wrapper that maps app.py's parameter names to generate_pdf_report."""
    return generate_pdf_report(
        patient=patient,
        result=result,
        findings=findings,
        doctor_name=doctor_name,
        heatmap_img=heatmap_arr,
        original_img=original_pil,
    )
