"""
🏥 MedAI Diagnostics — AI Medical Image Analyzer
Pages: Login | Dashboard | Analyze | Patients | History | Reports | Settings
"""
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime, date
import os, json, io

st.set_page_config(page_title="MedAI Diagnostics", page_icon="🏥",
                   layout="wide", initial_sidebar_state="expanded")

import auth, database as db
from utils import (get_css, load_image, clahe_enhance, np_to_pil,
                   image_to_bytes, auto_scan_type, confidence_bar,
                   sev_badge, build_findings, SEV_STYLE,
                   generate_patient_id)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = dict(user=None, page="dashboard", dark_mode=True,
                 result=None, heatmap=None, current_img=None)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

db.init_db()
auth.ensure_demo_account()
st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DB ADAPTER
# ─────────────────────────────────────────────────────────────────────────────
def _use_db():   return db.get_conn() is not None
def _patients(u): return db.get_patients(u) if _use_db() else db.mem_get_patients(u)
def _add_pat(d,u): return db.add_patient(d,u) if _use_db() else (db.mem_add_patient(d,u) or True)
def _get_pat(p):  return db.get_patient(p) if _use_db() else db.mem_get_patient(p)
def _save_scan(pid,st,fn,r,fi,uid):
    return db.save_scan(pid,st,fn,r,fi,uid) if _use_db() else db.mem_save_scan(pid,st,fn,r,fi,uid)
def _get_scans(p): return db.get_scans(p) if _use_db() else db.mem_get_scans(p)
def _all_scans(u): return db.get_all_scans(u) if _use_db() else db.mem_get_all_scans(u)
def _stats(u):    return db.get_dashboard_stats(u) if _use_db() else db.mem_get_stats(u)

def nav(page): st.session_state.page = page; st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# ── LOGIN PAGE ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def page_login():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1.5rem">
  <div style="font-size:3.5rem">🏥</div>
  <h1 style="font-size:2.2rem;margin:.5rem 0 .2rem;font-weight:800">MedAI Diagnostics</h1>
  <p style="color:#64748b;font-size:.9rem;margin:0">AI-Powered Medical Image Analysis Platform</p>
</div>""", unsafe_allow_html=True)

        tab_in, tab_up = st.tabs(["🔐 Sign In", "✨ Create Account"])

        with tab_in:
            with st.form("login"):
                u = st.text_input("Username", placeholder="Enter username")
                p = st.text_input("Password", type="password", placeholder="Enter password")
                go = st.form_submit_button("Sign In", use_container_width=True, type="primary")
            if go:
                ok, msg, user = auth.login(u, p)
                if ok: st.session_state.user = user; st.rerun()
                else: st.error(f"❌ {msg}")

            st.markdown("""
<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
  padding:12px 16px;margin-top:12px;font-size:.82rem;color:#94a3b8">
  <b style="color:#f1f5f9">🎯 Demo Credentials</b><br>
  Username: <code style="color:#60a5fa">demo_doctor</code> &nbsp;
  Password: <code style="color:#60a5fa">Demo@1234</code>
</div>""", unsafe_allow_html=True)

        with tab_up:
            with st.form("signup"):
                fn  = st.text_input("Full Name", placeholder="Dr. Jane Smith")
                nu  = st.text_input("Username",  placeholder="Choose a username")
                np_ = st.text_input("Password",  type="password", placeholder="Min 6 characters")
                np2 = st.text_input("Confirm Password", type="password")
                rl  = st.selectbox("Role", ["doctor","radiologist","nurse","admin"])
                reg = st.form_submit_button("Create Account", use_container_width=True, type="primary")
            if reg:
                if np_ != np2: st.error("❌ Passwords do not match.")
                else:
                    ok, msg = auth.signup(nu, np_, fn, rl)
                    if ok: st.success(f"✅ {msg} You can now sign in.")
                    else:  st.error(f"❌ {msg}")

        st.markdown("""
<p style="text-align:center;color:#475569;font-size:.72rem;margin-top:1.5rem">
For research &amp; educational use · Requires clinical validation
</p>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"""
<div style="padding:14px 4px 12px;border-bottom:1px solid #1e293b;margin-bottom:10px">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="width:44px;height:44px;border-radius:50%;background:linear-gradient(135deg,#2563eb,#1d4ed8);
      display:flex;align-items:center;justify-content:center;font-size:1.3rem;flex-shrink:0">👨‍⚕️</div>
    <div>
      <div style="font-weight:700;font-size:.9rem;color:#f1f5f9;line-height:1.2">
        {user.get('full_name') or user['username']}</div>
      <div style="font-size:.7rem;color:#475569;text-transform:capitalize">
        {user.get('role','doctor')}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        NAV = [("dashboard","📊","Dashboard"), ("analyze","🔬","Analyze Image"),
               ("patients","👥","Patients"),  ("history","📋","Scan History"),
               ("reports","📄","Reports"),    ("settings","⚙️","Settings")]

        cur = st.session_state.page
        for key, icon, label in NAV:
            active = cur == key
            bg  = "#2563eb" if active else "transparent"
            clr = "white"   if active else "#94a3b8"
            brd = "none"    if active else "1px solid transparent"
            st.markdown(f"""
<div style="background:{bg};color:{clr};border:{brd};border-radius:9px;
  padding:9px 12px;margin-bottom:3px;font-size:.85rem;font-weight:{'600' if active else '400'};
  display:flex;align-items:center;gap:9px">{icon} {label}</div>""",
                unsafe_allow_html=True)
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         label_visibility="collapsed"):
                nav(key)

        st.markdown("---")

        # Model status
        from model import weights_status, TORCH_OK
        st.markdown(f'<div style="font-size:.7rem;color:#475569;margin-bottom:6px">'
                    f'{"⚡ PyTorch ready" if TORCH_OK else "⚠️ Demo mode (no torch)"}</div>',
                    unsafe_allow_html=True)
        ws = weights_status()
        for scan_t, loaded in ws.items():
            dot = "🟢" if loaded else "🔴"
            short = scan_t.split()[0]
            st.markdown(f'<div style="font-size:.68rem;color:#64748b">{dot} {short}</div>',
                        unsafe_allow_html=True)

        st.markdown("---")
        dm_lbl = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
        if st.button(dm_lbl, use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode; st.rerun()

        db_lbl = "🟢 PostgreSQL" if _use_db() else "🟡 Demo (Memory)"
        st.markdown(f'<div style="font-size:.68rem;color:#64748b;padding:6px 8px;'
                    f'background:#1e293b;border-radius:7px;margin-top:6px">{db_lbl}</div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            auth.logout(); st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ── DASHBOARD ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard():
    user  = st.session_state.user
    stats = _stats(user["id"])
    scans = _all_scans(user["id"])

    st.markdown(f"## 📊 Dashboard")
    st.markdown(f'<p style="color:#64748b;margin-top:-10px">Welcome back, '
                f'<b>{user.get("full_name") or user["username"]}</b> · '
                f'{datetime.now():%A, %d %B %Y}</p>', unsafe_allow_html=True)

    # Metrics
    m1,m2,m3,m4 = st.columns(4)
    for col, val, label, icon, color in [
        (m1, stats["patients"], "Patients",       "👥", "#3b82f6"),
        (m2, stats["scans"],    "Total Scans",    "🔬", "#8b5cf6"),
        (m3, stats["critical"], "Critical Cases", "🔴", "#ef4444"),
        (m4, stats["reports"],  "Reports",        "📄", "#22c55e"),
    ]:
        with col:
            st.markdown(f'<div class="metric-box">'
                        f'<div class="metric-num" style="color:{color}">{icon} {val}</div>'
                        f'<div class="metric-lbl">{label}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    col_rec, col_dist = st.columns([3, 2])

    with col_rec:
        st.markdown("#### 🕐 Recent Scans")
        if scans:
            st.markdown('<table class="scan-table"><thead><tr>'
                        '<th>Patient</th><th>Type</th><th>Severity</th><th>Date</th>'
                        '</tr></thead><tbody>', unsafe_allow_html=True)
            for s in scans[:10]:
                sev = s.get("severity","—")
                st.markdown(f'<tr>'
                            f'<td><b>{s.get("patient_name","—")}</b></td>'
                            f'<td>{s.get("scan_type","—")}</td>'
                            f'<td>{sev_badge(sev)}</td>'
                            f'<td>{str(s.get("created_at",""))[:10]}</td>'
                            f'</tr>', unsafe_allow_html=True)
            st.markdown('</tbody></table>', unsafe_allow_html=True)
        else:
            st.info("No scans yet. Go to **Analyze Image** to get started.")
            if st.button("🔬 Start Analyzing", type="primary"): nav("analyze")

    with col_dist:
        st.markdown("#### 📈 Severity Breakdown")
        if scans:
            cnts = {}
            for s in scans: cnts[s.get("severity","Unknown")] = cnts.get(s.get("severity","Unknown"),0)+1
            order = ["Critical","High","Moderate","Low","Normal","Unknown"]
            clrs  = {"Critical":"#ef4444","High":"#f97316","Moderate":"#eab308",
                     "Low":"#22c55e","Normal":"#3b82f6","Unknown":"#94a3b8"}
            for sv in order:
                if sv not in cnts: continue
                cnt = cnts[sv]; pct = int(cnt/len(scans)*100)
                c   = clrs.get(sv,"#94a3b8")
                st.markdown(
                    f'<div style="margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:3px">'
                    f'<span style="font-weight:600">{sv}</span>'
                    f'<span style="color:{c}">{cnt} ({pct}%)</span></div>'
                    f'<div style="background:#334155;border-radius:4px;height:7px">'
                    f'<div style="background:{c};width:{pct}%;height:7px;border-radius:4px"></div>'
                    f'</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#64748b;font-size:.85rem">No data yet.</p>',
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ⚡ Quick Actions")
    q1,q2,q3,q4 = st.columns(4)
    with q1:
        if st.button("🔬 New Analysis", use_container_width=True, type="primary"): nav("analyze")
    with q2:
        if st.button("➕ Add Patient",  use_container_width=True): nav("patients")
    with q3:
        if st.button("📋 Scan History", use_container_width=True): nav("history")
    with q4:
        if st.button("📄 Reports",      use_container_width=True): nav("reports")


# ─────────────────────────────────────────────────────────────────────────────
# ── ANALYZE IMAGE ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def page_analyze():
    from model import predict, generate_gradcam

    user = st.session_state.user
    st.markdown("## 🔬 Analyze Medical Image")

    col_up, col_res = st.columns([1, 1])

    # ── Upload panel ─────────────────────────────────────────────────────────
    with col_up:
        with st.container():
            st.markdown("#### 📤 Upload & Configure")

            scan_type = st.selectbox("Scan Type", ["Chest X-Ray","Brain MRI","CT Scan"])

            pats = _patients(user["id"])
            pat_map = {f"{p['name']} ({p['patient_id']})": p["patient_id"] for p in pats}
            pat_sel = st.selectbox("Patient", ["— Select patient —"] + list(pat_map.keys()))
            patient_id = pat_map.get(pat_sel)

            uploaded = st.file_uploader(
                "Upload medical image",
                type=["png","jpg","jpeg","bmp","tiff"],
                label_visibility="visible")

            use_sample = False
            SAMPLES = {
                "Chest X-Ray": "data/sample_images/chest_xray_sample.png",
                "Brain MRI":   "data/sample_images/brain_mri_sample.png",
                "CT Scan":     "data/sample_images/ct_scan_sample.png",
            }
            if not uploaded:
                st.markdown('<p style="color:#64748b;font-size:.78rem;margin:6px 0">or use a demo:</p>',
                            unsafe_allow_html=True)
                if st.button("📷 Load Sample Image", use_container_width=True):
                    use_sample = True
                    st.session_state["_sample_type"] = scan_type

            apply_clahe = st.checkbox("🔧 CLAHE Enhancement", value=True,
                                      help="Contrast Limited Adaptive Histogram Equalization — standard medical pre-processing")

        # Load image
        pil_img = None
        if uploaded:
            pil_img   = load_image(uploaded)
            scan_type = auto_scan_type(uploaded.name)
        elif use_sample or st.session_state.get("_sample_type"):
            sp = SAMPLES.get(st.session_state.get("_sample_type", scan_type))
            if sp and os.path.exists(sp):
                pil_img = load_image(sp)

        if pil_img:
            display = clahe_enhance(pil_img) if apply_clahe else pil_img
            st.image(display, caption=f"{scan_type} Preview ({pil_img.width}×{pil_img.height}px)",
                     use_container_width=True)

            if st.button("🧠 Run AI Analysis", type="primary", use_container_width=True):
                with st.spinner("Analysing image + generating heatmap…"):
                    result  = predict(pil_img, scan_type)
                    heatmap = generate_gradcam(pil_img, scan_type, result)
                    st.session_state.result      = result
                    st.session_state.heatmap     = heatmap
                    st.session_state.current_img = pil_img
                st.rerun()

    # ── Results panel ─────────────────────────────────────────────────────────
    with col_res:
        result  = st.session_state.result
        heatmap = st.session_state.heatmap
        pil_img = st.session_state.current_img

        if result is None:
            st.markdown("""
<div style="height:420px;display:flex;flex-direction:column;align-items:center;
  justify-content:center;background:#1e293b;border-radius:14px;
  border:2px dashed #334155;color:#475569">
  <div style="font-size:3rem">🏥</div>
  <div style="font-size:1rem;margin-top:.8rem;font-weight:600">Upload an image to see results</div>
  <div style="font-size:.78rem;margin-top:.4rem">Supports Chest X-Ray · Brain MRI · CT Scan</div>
</div>""", unsafe_allow_html=True)
            return

        sev   = result.get("severity","Unknown")
        ss    = SEV_STYLE.get(sev, SEV_STYLE["Low"])
        model = result.get("model","AI Model")
        demo  = result.get("demo", True)

        # Severity banner
        st.markdown(
            f'<div style="background:{ss["bg"]};border:2px solid {ss["border"]};'
            f'border-radius:12px;padding:14px 18px;margin-bottom:12px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div><span style="font-size:1.6rem">{ss["icon"]}</span>'
            f'<span style="font-size:1.1rem;font-weight:800;color:{ss["text"]};margin-left:10px">'
            f'{sev.upper()}</span></div>'
            f'<span style="font-size:.7rem;color:#64748b;max-width:55%;text-align:right">{model}</span></div>'
            f'<div style="font-size:.8rem;color:{ss["text"]};margin-top:4px">'
            f'{result.get("type","")}'
            f'{"  ·  ⚠️ Demo mode" if demo else "  ·  ✅ Trained weights"}</div></div>',
            unsafe_allow_html=True)

        t_hm, t_prob, t_find = st.tabs(["🌡️ Heatmap","📊 Probabilities","📝 Findings"])

        with t_hm:
            if heatmap is not None:
                c1,c2 = st.columns(2)
                with c1: st.image(pil_img, caption="Original",     use_container_width=True)
                with c2: st.image(np_to_pil(heatmap), caption="Grad-CAM", use_container_width=True)
                st.markdown('<p style="font-size:.75rem;color:#64748b">'
                            '🔥 Red = highest model attention. Used for AI explainability.</p>',
                            unsafe_allow_html=True)
            else:
                if pil_img:
                    st.image(pil_img, caption="Original Image", use_container_width=True)
                st.info("Grad-CAM unavailable — install PyTorch to enable heatmaps.")

        with t_prob:
            preds  = result.get("predictions",{})
            sorted_p = sorted(preds.items(), key=lambda x:-x[1])
            color_fn = lambda p: ("#ef4444" if p>.6 else "#f97316" if p>.4
                                  else "#eab308" if p>.25 else "#22c55e")
            for cond, prob in sorted_p:
                st.markdown(confidence_bar(cond, prob, color_fn(prob)),
                            unsafe_allow_html=True)

        with t_find:
            auto_txt = build_findings(result)
            findings = st.text_area("AI Findings (editable before saving):",
                                    auto_txt, height=160, key="find_edit")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            can_save = bool(patient_id)
            if st.button("💾 Save to Patient Record",
                         use_container_width=True, type="primary",
                         disabled=not can_save):
                fname = f"{scan_type.replace(' ','_')}_{datetime.now():%Y%m%d_%H%M%S}.png"
                sid   = _save_scan(patient_id, scan_type, fname, result,
                                   st.session_state.get("find_edit", auto_txt), user["id"])
                if sid: st.success(f"✅ Saved! Scan ID: {sid}")
            if not can_save:
                st.caption("Select a patient to enable saving.")

        with c2:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                pat = _get_patient(patient_id) if patient_id else \
                      {"name":"Demo Patient","patient_id":"DEMO-001",
                       "dob":"N/A","gender":"N/A","blood_type":"N/A"}
                from report import generate_pdf_report
                pdf = generate_pdf_report(
                    patient     = pat or {"name":"Unknown","patient_id":"N/A"},
                    result      = result,
                    findings    = st.session_state.get("find_edit", auto_txt),
                    doctor_name = user.get("full_name") or user["username"],
                    heatmap_img = heatmap,
                    original_img= pil_img,
                )
                fname = f"MedAI_{datetime.now():%Y%m%d_%H%M%S}.pdf"
                st.download_button("⬇️ Download PDF", pdf, fname,
                                   "application/pdf", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── PATIENTS ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def page_patients():
    user = st.session_state.user
    st.markdown("## 👥 Patient Management")

    tab_list, tab_add = st.tabs(["📋 Patient List","➕ Add Patient"])

    with tab_list:
        pats = _patients(user["id"])
        if not pats:
            st.info("No patients yet. Add your first patient in the ➕ Add Patient tab.")
        else:
            q = st.text_input("🔍 Search", placeholder="Name or ID…")
            if q:
                ql = q.lower()
                pats = [p for p in pats
                        if ql in p.get("name","").lower()
                        or ql in p.get("patient_id","").lower()]

            st.markdown('<table class="scan-table"><thead><tr>'
                        '<th>ID</th><th>Name</th><th>DOB</th>'
                        '<th>Gender</th><th>Blood</th></tr></thead><tbody>',
                        unsafe_allow_html=True)
            for p in pats:
                st.markdown(
                    f'<tr><td><code>{p["patient_id"]}</code></td>'
                    f'<td><b>{p["name"]}</b></td>'
                    f'<td>{str(p.get("dob","—"))[:10]}</td>'
                    f'<td>{p.get("gender","—")}</td>'
                    f'<td>{p.get("blood_type","—")}</td></tr>',
                    unsafe_allow_html=True)
            st.markdown('</tbody></table>', unsafe_allow_html=True)

            if pats:
                sel = st.selectbox("View patient details:", [p["patient_id"] for p in pats])
                if sel:
                    p = _get_patient(sel)
                    if p:
                        scans = _get_scans(sel)
                        with st.expander(f"📋 {p['name']} — Full Record", expanded=True):
                            ca, cb = st.columns(2)
                            with ca:
                                for k,lbl in [("patient_id","ID"),("dob","DOB"),
                                               ("gender","Gender"),("blood_type","Blood")]:
                                    st.markdown(f"**{lbl}:** {p.get(k,'—')}")
                            with cb:
                                for k,lbl in [("phone","Phone"),("email","Email"),("notes","Notes")]:
                                    st.markdown(f"**{lbl}:** {p.get(k,'—')}")
                            st.markdown(f"**Scans:** {len(scans)}")
                            for s in scans[:5]:
                                st.markdown(
                                    f'`{str(s.get("created_at",""))[:10]}` '
                                    f'**{s.get("scan_type","—")}** — '
                                    f'{sev_badge(s.get("severity","—"))}',
                                    unsafe_allow_html=True)

    with tab_add:
        st.markdown("#### Register New Patient")
        with st.form("add_patient"):
            c1, c2 = st.columns(2)
            with c1:
                name   = st.text_input("Full Name *", placeholder="John Doe")
                dob    = st.date_input("Date of Birth", value=date(1990,1,1),
                                       min_value=date(1900,1,1))
                gender = st.selectbox("Gender", ["Male","Female","Other","Prefer not to say"])
            with c2:
                pid    = st.text_input("Patient ID (blank = auto)",
                                       placeholder=generate_patient_id())
                blood  = st.selectbox("Blood Type",
                                      ["A+","A−","B+","B−","AB+","AB−","O+","O−","Unknown"])
                phone  = st.text_input("Phone", placeholder="+1-555-0000")
            email = st.text_input("Email", placeholder="patient@example.com")
            notes = st.text_area("Clinical Notes", height=80,
                                 placeholder="Allergies, conditions, medications…")
            add_btn = st.form_submit_button("➕ Register Patient",
                                            use_container_width=True, type="primary")
        if add_btn:
            if not name: st.error("Patient name is required.")
            else:
                d = {"patient_id": pid.strip() or generate_patient_id(),
                     "name":name,"dob":str(dob),"gender":gender,
                     "blood_type":blood,"phone":phone,"email":email,"notes":notes}
                if _add_pat(d, user["id"]):
                    st.success(f"✅ '{name}' registered! ID: {d['patient_id']}")
                else:
                    st.error("Registration failed.")


# ─────────────────────────────────────────────────────────────────────────────
# ── HISTORY ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def page_history():
    user  = st.session_state.user
    scans = _all_scans(user["id"])
    st.markdown("## 📋 Scan History")

    if not scans:
        st.info("No scans yet.")
        if st.button("🔬 Start Analyzing"): nav("analyze")
        return

    c1,c2,c3 = st.columns(3)
    with c1: ft = st.selectbox("Type", ["All","Chest X-Ray","Brain MRI","CT Scan"])
    with c2: fs = st.selectbox("Severity", ["All","Critical","High","Moderate","Low","Normal"])
    with c3: fq = st.text_input("Search patient")

    f = scans
    if ft != "All": f = [s for s in f if s.get("scan_type")==ft]
    if fs != "All": f = [s for s in f if s.get("severity")==fs]
    if fq: ql = fq.lower(); f = [s for s in f if ql in s.get("patient_name","").lower()]

    st.markdown(f'<p style="color:#64748b;font-size:.8rem">{len(f)} of {len(scans)} scans</p>',
                unsafe_allow_html=True)

    st.markdown('<table class="scan-table"><thead><tr>'
                '<th>#</th><th>Patient</th><th>Type</th>'
                '<th>Severity</th><th>Preview</th><th>Date</th>'
                '</tr></thead><tbody>', unsafe_allow_html=True)
    for i,s in enumerate(f[:50],1):
        sev = s.get("severity","—")
        prev= str(s.get("findings",""))[:55]+"…" if s.get("findings") else "—"
        st.markdown(
            f'<tr><td>{i}</td>'
            f'<td><b>{s.get("patient_name","—")}</b></td>'
            f'<td>{s.get("scan_type","—")}</td>'
            f'<td>{sev_badge(sev)}</td>'
            f'<td style="font-size:.75rem;color:#94a3b8">{prev}</td>'
            f'<td>{str(s.get("created_at",""))[:10]}</td></tr>',
            unsafe_allow_html=True)
    st.markdown('</tbody></table>', unsafe_allow_html=True)

    if f:
        df = pd.DataFrame([{"Patient":s.get("patient_name",""),
                            "Scan Type":s.get("scan_type",""),
                            "Severity":s.get("severity",""),
                            "Findings":str(s.get("findings",""))[:80],
                            "Date":str(s.get("created_at",""))[:10]}
                           for s in f])
        st.download_button("⬇️ Export CSV", df.to_csv(index=False),
                           "scan_history.csv","text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# ── REPORTS ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def page_reports():
    from report import generate_pdf_report
    user = st.session_state.user
    st.markdown("## 📄 Generate Reports")

    pats = _patients(user["id"])
    if not pats:
        st.info("Add patients first, then analyze scans to generate reports.")
        return

    pm = {f"{p['name']} ({p['patient_id']})": p["patient_id"] for p in pats}
    sel_p = st.selectbox("Patient:", list(pm.keys()))
    pid   = pm[sel_p]
    pat   = _get_patient(pid)
    scans = _get_scans(pid)

    if not scans:
        st.info(f"No scans for this patient yet — analyze an image first.")
        return

    sm = {f"{s.get('scan_type','')} — {s.get('severity','')} [{str(s.get('created_at',''))[:10]}]": s
          for s in scans}
    sel_s = st.selectbox("Scan:", list(sm.keys()))
    scan  = sm[sel_s]
    res   = scan.get("result_json",{})
    if isinstance(res, str):
        try: res = json.loads(res)
        except: res = {}

    findings = scan.get("findings","") or build_findings(res)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Patient Info**")
        for k,l in [("name","Name"),("patient_id","ID"),("dob","DOB"),("gender","Gender")]:
            st.markdown(f'<span style="font-size:.82rem"><b>{l}:</b> {pat.get(k,"—")}</span><br>',
                        unsafe_allow_html=True)
    with cb:
        st.markdown("**Scan Info**")
        st.markdown(sev_badge(res.get("severity","—")), unsafe_allow_html=True)
        for k,l in [("type","Type"),("model","Model")]:
            st.markdown(f'<span style="font-size:.82rem"><b>{l}:</b> {res.get(k,"—")}</span><br>',
                        unsafe_allow_html=True)

    findings_edit = st.text_area("Edit Findings:", findings, height=120)

    if st.button("📄 Generate & Download PDF", type="primary", use_container_width=True):
        pdf = generate_pdf_report(
            patient     = pat,
            result      = res,
            findings    = findings_edit,
            doctor_name = user.get("full_name") or user["username"],
        )
        fn = f"MedAI_{pat.get('patient_id','')}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        st.download_button("⬇️ Download PDF", pdf, fn, "application/pdf",
                           use_container_width=True)
        st.success("✅ PDF generated!")


# ─────────────────────────────────────────────────────────────────────────────
# ── SETTINGS ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def page_settings():
    from model import weights_status, TORCH_OK
    from report import REPORTLAB_OK

    user = st.session_state.user
    st.markdown("## ⚙️ Settings")

    t1, t2, t3, t4 = st.tabs(["👤 Profile","🗄️ Database","🤖 Models","📖 Dataset Guide"])

    with t1:
        st.markdown("#### Profile")
        with st.form("profile"):
            nn = st.text_input("Full Name", value=user.get("full_name",""))
            if st.form_submit_button("Save", type="primary"):
                st.success("✅ Saved.")
        st.markdown("---")
        st.markdown("#### Theme")
        if st.button("Toggle Dark/Light Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode; st.rerun()

    with t2:
        st.markdown("#### Neon PostgreSQL Setup")
        ok = _use_db()
        st.markdown(
            f'<div style="padding:12px;background:{"#052e16" if ok else "#1c1917"};'
            f'border:1px solid {"#16a34a" if ok else "#a16207"};border-radius:10px;'
            f'margin-bottom:12px">'
            f'<b>{"🟢 Connected — data persists across sessions" if ok else "🟡 In-Memory mode — data resets on restart"}</b>'
            f'</div>', unsafe_allow_html=True)
        st.code("""# Streamlit Cloud → App Settings → Secrets:
DATABASE_URL = "postgresql://neondb_owner:PASSWORD@ep-XXXX.neon.tech/neondb?sslmode=require"

# Or locally in .streamlit/secrets.toml
DATABASE_URL = "postgresql://..."
""")
        st.markdown("Get your connection string from **[neon.tech](https://neon.tech)** → Connect")

    with t3:
        st.markdown("#### AI Model Status")
        ws = weights_status()
        for scan_t, loaded in ws.items():
            st.markdown(
                f'<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;'
                f'padding:12px 16px;margin-bottom:8px">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<b>{scan_t}</b>'
                f'<span style="padding:3px 10px;border-radius:999px;font-size:.72rem;font-weight:700;'
                f'background:{"#052e16" if loaded else "#1c1917"};'
                f'color:{"#4ade80" if loaded else "#fbbf24"}">'
                f'{"✅ Loaded" if loaded else "🔄 Demo Mode"}</span></div></div>',
                unsafe_allow_html=True)
        st.markdown(f"PyTorch: **{'✅ Available' if TORCH_OK else '❌ Not installed'}**  "
                    f"| ReportLab: **{'✅' if REPORTLAB_OK else '❌'}**")
        st.code("""# After training, weights auto-load from:
weights/chest_xray.pth   ← python train_chest.py
weights/brain_mri.pth    ← python train_brain.py
""")

    with t4:
        st.markdown("#### Dataset Guide")
        st.markdown("""
**Option A — Download real datasets (recommended):**
```bash
python download_data.py --dataset brain    # ~150 MB Brain MRI
python download_data.py --dataset chest    # ~1.2 GB Chest X-Ray Pneumonia
python download_data.py --notebook brain   # 99% accuracy reference notebook
```

**Option B — Generate synthetic training data (no download):**
```bash
python generate_synthetic.py               # all datasets
python generate_synthetic.py --type brain  # brain MRI only
python generate_synthetic.py --type chest  # chest X-ray only
python generate_synthetic.py --samples 300 # 300 images/class
```

**Then train the models:**
```bash
python train_brain.py   --epochs 20   # → weights/brain_mri.pth
python train_chest.py   --epochs 10   # → weights/chest_xray.pth
```
""")


# ─────────────────────────────────────────────────────────────────────────────
# ── ROUTER ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not auth.is_logged_in():
        page_login(); return

    render_sidebar()
    page = st.session_state.page
    dispatch = {
        "dashboard": page_dashboard,
        "analyze":   page_analyze,
        "patients":  page_patients,
        "history":   page_history,
        "reports":   page_reports,
        "settings":  page_settings,
    }
    dispatch.get(page, page_dashboard)()

if __name__ == "__main__":
    main()
