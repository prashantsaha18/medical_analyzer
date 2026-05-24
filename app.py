"""
MedAI Diagnostics — AI Medical Image Analyzer
7 pages: Login | Dashboard | Analyze | Patients | History | Reports | Settings
All Streamlit widget labels are non-empty (fixes v1.57+ warnings)
"""
import json
import os
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="MedAI Diagnostics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

import auth
import database as db
from model import predict, generate_gradcam, weights_status, TORCH_AVAILABLE
from report import generate_pdf, RL_OK
from utils import (
    get_css, load_image, clahe_enhance, np_to_pil,
    pil_to_bytes, auto_detect_scan_type,
    confidence_bar, sev_badge, build_findings,
    SEV_STYLES, generate_patient_id,
)

# ── Startup ────────────────────────────────────────────────────────────────────
db.init_db()

# ── Session defaults ──────────────────────────────────────────────────────────
_DEFAULTS = {
    "user":          None,
    "page":          "dashboard",
    "dark_mode":     True,
    "result":        None,
    "heatmap":       None,
    "current_img":   None,
    "_sample_type":  None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)


# ── Navigation helper ─────────────────────────────────────────────────────────
def _go(page: str) -> None:
    st.session_state.page = page
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ── LOGIN PAGE ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def page_login() -> None:
    _, col, _ = st.columns([1, 1.8, 1])
    with col:
        st.markdown(
            """
<div class="float-card" style="text-align:center;padding:3rem 0 2rem;margin-bottom:1rem">
  <div class="pulse-icon" style="font-size:4.5rem;margin-bottom:0.8rem">🏥</div>
  <h1 style="font-size:2.8rem;margin:0;font-weight:800;background:linear-gradient(135deg,#388bfd,#8957e5);-webkit-background-clip:text;-webkit-text-fill-color:transparent">MedAI Diagnostics</h1>
  <p style="color:#8b949e;font-size:1.02rem;margin-top:0.6rem;font-weight:400;letter-spacing:0.5px">Next-Generation AI Clinical Image Analytics</p>
</div>""",
            unsafe_allow_html=True,
        )

        tab_in, tab_up = st.tabs(["🔐 Sign In", "✨ Create Account"])

        with tab_in:
            with st.form("form_login"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password",
                                          placeholder="Enter your password")
                submitted = st.form_submit_button(
                    "Sign In", use_container_width=True, type="primary"
                )
            if submitted:
                ok, msg, user = auth.login(username, password)
                if ok:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

        with tab_up:
            with st.form("form_signup"):
                full_name = st.text_input("Full Name",   placeholder="Dr. Jane Smith")
                new_user  = st.text_input("Username",    placeholder="Choose a username (min 3 chars)")
                new_pass  = st.text_input("Password",    type="password",
                                           placeholder="Minimum 6 characters")
                new_pass2 = st.text_input("Confirm Password", type="password")
                role      = st.selectbox("Role", ["doctor", "radiologist", "nurse", "admin"])
                register  = st.form_submit_button(
                    "Create Account", use_container_width=True, type="primary"
                )
            if register:
                if new_pass != new_pass2:
                    st.error("❌ Passwords do not match.")
                else:
                    ok, msg = auth.signup(new_user, new_pass, full_name, role)
                    if ok:
                        st.success(f"✅ {msg} You can now sign in.")
                    else:
                        st.error(f"❌ {msg}")

        st.markdown(
            '<p style="text-align:center;color:#464f58;font-size:.78rem;margin-top:1.8rem">'
            "For research and educational use · All findings require clinical validation"
            "</p>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    user = st.session_state.user
    with st.sidebar:
        # User info
        st.markdown(
            f"""
<div style="padding:12px 4px 10px;border-bottom:1px solid #21262d;margin-bottom:8px">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="width:42px;height:42px;border-radius:50%;
      background:linear-gradient(135deg,#1f6feb,#388bfd);
      display:flex;align-items:center;justify-content:center;
      font-size:1.25rem;flex-shrink:0">👨‍⚕️</div>
    <div>
      <div style="font-weight:700;font-size:.88rem;color:#e6edf3;line-height:1.2">
        {user.get("full_name") or user["username"]}</div>
      <div style="font-size:.68rem;color:#484f58;text-transform:capitalize">
        {user.get("role", "doctor")}</div>
    </div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        # Navigation — plain st.button, no label_visibility (not supported on buttons)
        NAV = [
            ("dashboard", "📊", "Dashboard"),
            ("analyze",   "🔬", "Analyze Image"),
            ("patients",  "👥", "Patients"),
            ("history",   "📋", "Scan History"),
            ("reports",   "📄", "Reports"),
            ("settings",  "⚙️", "Settings"),
        ]
        cur = st.session_state.page
        for key, icon, label in NAV:
            active = cur == key
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                _go(key)

        st.markdown("---")

        # Model status
        ws = weights_status()
        st.markdown(
            f'<div style="font-size:.68rem;color:#484f58;margin-bottom:4px">'
            f'{"⚡ PyTorch ready" if TORCH_AVAILABLE else "⚠ No PyTorch"}</div>',
            unsafe_allow_html=True,
        )
        for scan_t, loaded in ws.items():
            dot = "🟢" if loaded else "🔴"
            st.markdown(
                f'<div style="font-size:.68rem;color:#636e7b">'
                f'{dot} {scan_t.split()[0]} weights</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        dm_lbl = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
        if st.button(dm_lbl, use_container_width=True, key="btn_theme"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

        db_status = "🟢 PostgreSQL (Production)" if db.is_db_available() else "⚠️ Local Database (Demo Fallback)"
        st.markdown(
            f'<div style="font-size:.68rem;color:#8b949e;padding:6px 8px;'
            f'background:#161b22;border-radius:7px;margin-top:6px">{db_status}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        if st.button("🚪 Sign Out", use_container_width=True, key="btn_logout"):
            auth.logout()
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ── DASHBOARD ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard() -> None:
    user  = st.session_state.user
    stats = db.db_get_stats(user["id"])
    scans = db.db_get_all_scans(user["id"])

    st.markdown("## 📊 Dashboard")
    st.markdown(
        f'<p style="color:#8b949e;margin-top:-8px">'
        f'Welcome back, <b>{user.get("full_name") or user["username"]}</b> · '
        f'{datetime.now():%A, %d %B %Y}</p>',
        unsafe_allow_html=True,
    )

    if not db.is_db_available():
        st.warning("⚠️ **System Status: Local Demo Fallback Mode** · The `DATABASE_URL` environment variable is not configured. Scans and patient data are currently stored in-memory and will be cleared when the server restarts. Please configure a PostgreSQL database for permanent production storage.")

    # Metric row
    m1, m2, m3, m4 = st.columns(4)
    for col, val, label, icon, color in (
        (m1, stats["patients"], "Patients",       "👥", "#1f6feb"),
        (m2, stats["scans"],    "Total Scans",    "🔬", "#8957e5"),
        (m3, stats["critical"], "Critical Cases", "🔴", "#f85149"),
        (m4, stats["reports"],  "Reports",        "📄", "#3fb950"),
    ):
        with col:
            st.markdown(
                f'<div class="mbox"><div class="mbox-num" style="color:{color}">'
                f'{icon} {val}</div><div class="mbox-lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    col_rec, col_dist = st.columns([3, 2])

    with col_rec:
        st.markdown("#### 🕐 Recent Scans")
        if scans:
            st.markdown(
                '<table class="med-table"><thead><tr>'
                "<th>Patient</th><th>Type</th><th>Severity</th><th>Date</th>"
                "</tr></thead><tbody>",
                unsafe_allow_html=True,
            )
            for s in scans[:10]:
                sev = s.get("severity", "—")
                dt  = str(s.get("created_at", ""))[:10]
                st.markdown(
                    f"<tr>"
                    f"<td><b>{s.get('patient_name','—')}</b></td>"
                    f"<td>{s.get('scan_type','—')}</td>"
                    f"<td>{sev_badge(sev)}</td>"
                    f"<td>{dt}</td></tr>",
                    unsafe_allow_html=True,
                )
            st.markdown("</tbody></table>", unsafe_allow_html=True)
        else:
            st.info("No scans yet. Go to **Analyze Image** to get started.")
            if st.button("🔬 Start Analyzing", type="primary", key="dash_start"):
                _go("analyze")

    with col_dist:
        st.markdown("#### 📈 Severity Breakdown")
        if scans:
            order = ["Critical", "High", "Moderate", "Low", "Normal"]
            clrs  = {
                "Critical": "#f85149", "High": "#f97316",
                "Moderate": "#e3b341", "Low": "#3fb950",
                "Normal":   "#1f6feb",
            }
            cnts = {}
            for s in scans:
                sv = s.get("severity", "Unknown")
                cnts[sv] = cnts.get(sv, 0) + 1
            for sv in order:
                if sv not in cnts:
                    continue
                cnt = cnts[sv]
                pct = int(cnt / len(scans) * 100)
                c   = clrs.get(sv, "#636e7b")
                st.markdown(
                    f'<div style="margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:.8rem;margin-bottom:3px">'
                    f'<span style="font-weight:500">{sv}</span>'
                    f'<span style="color:{c}">{cnt} ({pct}%)</span></div>'
                    f'<div style="background:#21262d;border-radius:4px;height:7px">'
                    f'<div style="background:{c};width:{pct}%;height:7px;'
                    f'border-radius:4px"></div></div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<p style="color:#636e7b;font-size:.85rem">No data yet.</p>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### ⚡ Quick Actions")
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("🔬 New Analysis", use_container_width=True,
                     type="primary", key="qa_analyze"):
            _go("analyze")
    with qa2:
        if st.button("➕ Add Patient", use_container_width=True, key="qa_patient"):
            _go("patients")
    with qa3:
        if st.button("📋 Scan History", use_container_width=True, key="qa_history"):
            _go("history")
    with qa4:
        if st.button("📄 Reports", use_container_width=True, key="qa_reports"):
            _go("reports")


# ─────────────────────────────────────────────────────────────────────────────
# ── ANALYZE IMAGE ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def page_analyze() -> None:
    user = st.session_state.user
    st.markdown("## 🔬 Analyze Medical Image")

    col_up, col_res = st.columns([1, 1])

    # ── Upload panel ──────────────────────────────────────────────────────────
    with col_up:
        st.markdown("#### 📤 Upload & Configure")

        scan_type = st.selectbox(
            "Scan Type", ["Chest X-Ray", "Brain MRI", "CT Scan"],
            key="sel_scan_type",
        )

        pats    = db.db_get_patients(user["id"])
        pat_map = {f"{p['name']} ({p['patient_id']})": p["patient_id"] for p in pats}
        pat_sel = st.selectbox(
            "Assign to Patient",
            ["— select patient —"] + list(pat_map.keys()),
            key="sel_patient",
        )
        patient_id = pat_map.get(pat_sel)

        uploaded = st.file_uploader(
            "Upload medical image (PNG / JPG / BMP / TIFF)",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
        )

        # Sample image button
        _SAMPLES = {
            "Chest X-Ray": "data/sample_images/chest_xray_sample.png",
            "Brain MRI":   "data/sample_images/brain_mri_sample.png",
            "CT Scan":     "data/sample_images/ct_scan_sample.png",
        }
        if not uploaded:
            st.caption("No file chosen — use a built-in demo image:")
            if st.button("📷 Load Sample Image", key="btn_sample"):
                st.session_state["_sample_type"] = scan_type

        apply_clahe = st.checkbox(
            "🔧 Apply CLAHE Enhancement",
            value=True,
            help="Contrast Limited Adaptive Histogram Equalisation — "
                 "standard pre-processing for medical images",
        )

        # Resolve image source
        pil_img = None
        if uploaded:
            pil_img   = load_image(uploaded)
            scan_type = auto_detect_scan_type(uploaded.name)
        elif st.session_state.get("_sample_type"):
            sp = _SAMPLES.get(st.session_state["_sample_type"], _SAMPLES["Chest X-Ray"])
            if os.path.isfile(sp):
                pil_img = load_image(sp)

        if pil_img is not None:
            display_img = clahe_enhance(pil_img) if apply_clahe else pil_img
            st.image(
                display_img,
                caption=f"{scan_type} · {pil_img.width}×{pil_img.height}px",
                use_container_width=True,
            )

            if st.button(
                "🧠 Run AI Analysis",
                type="primary",
                use_container_width=True,
                key="btn_run",
            ):
                with st.spinner("Running inference + generating Grad-CAM++…"):
                    result  = predict(pil_img, scan_type)
                    heatmap = generate_gradcam(pil_img, scan_type, result)
                    st.session_state.result      = result
                    st.session_state.heatmap     = heatmap
                    st.session_state.current_img = pil_img
                st.rerun()

    # ── Results panel ──────────────────────────────────────────────────────────
    with col_res:
        result  = st.session_state.result
        heatmap = st.session_state.heatmap
        pil_img = st.session_state.current_img

        if result is None:
            st.markdown(
                """
<div style="height:420px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  background:#161b22;border-radius:12px;border:2px dashed #30363d;
  color:#484f58">
  <div style="font-size:3rem">🏥</div>
  <div style="font-size:1rem;margin-top:.8rem;font-weight:600">
    Upload an image to see AI results</div>
  <div style="font-size:.78rem;margin-top:.3rem">
    Chest X-Ray · Brain MRI · CT Scan</div>
</div>""",
                unsafe_allow_html=True,
            )
            return

        sev   = result.get("severity", "Low")
        ss    = SEV_STYLES.get(sev, SEV_STYLES["Low"])
        is_demo = result.get("demo", True)

        # Severity banner
        st.markdown(
            f'<div style="background:{ss["bg"]};border:2px solid {ss["border"]};'
            f'border-radius:12px;padding:14px 18px;margin-bottom:12px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div><span style="font-size:1.5rem">{ss["icon"]}</span>'
            f'<span style="font-size:1rem;font-weight:800;color:{ss["text"]};'
            f'margin-left:10px">{sev.upper()}</span></div>'
            f'<span style="font-size:.68rem;color:#636e7b">'
            f'{result.get("model","")}</span></div>'
            f'<div style="font-size:.78rem;color:{ss["text"]};margin-top:4px">'
            f'{result.get("type","")} · '
            f'{"ℹ️ Simulated Inference" if is_demo else "✅ Trained Model"}</div></div>',
            unsafe_allow_html=True,
        )

        t_hm, t_prob, t_find = st.tabs(
            ["🌡️ Grad-CAM++", "📊 Probabilities", "📝 Findings"]
        )

        with t_hm:
            if heatmap is not None and pil_img is not None:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(pil_img,          caption="Original",       use_container_width=True)
                with c2:
                    st.image(np_to_pil(heatmap), caption="Grad-CAM++ Map", use_container_width=True)
                st.caption(
                    "🔥 Red zones = highest model attention. "
                    "Used for AI explainability and clinical review."
                )
            else:
                if pil_img:
                    st.image(pil_img, caption="Original Image", use_container_width=True)
                st.info(
                    "Heatmap requires PyTorch. "
                    "Install it with: pip install torch torchvision"
                )

        with t_prob:
            preds = result.get("predictions", {})
            sorted_p = sorted(preds.items(), key=lambda x: -x[1])
            for label, prob in sorted_p:
                c = (
                    "#f85149" if prob > 0.6 else
                    "#f97316" if prob > 0.4 else
                    "#e3b341" if prob > 0.25 else
                    "#3fb950"
                )
                st.markdown(confidence_bar(label, prob, c), unsafe_allow_html=True)

        with t_find:
            auto_txt = build_findings(result)
            findings_edit = st.text_area(
                "AI Findings (editable before saving):",
                auto_txt,
                height=165,
                key="findings_textarea",
            )

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            can_save = bool(patient_id)
            if st.button(
                "💾 Save to Record",
                use_container_width=True,
                type="primary",
                key="btn_save",
                disabled=not can_save,
            ):
                fname = (
                    f"{scan_type.replace(' ', '_')}"
                    f"_{datetime.now():%Y%m%d_%H%M%S}.png"
                )
                sid = db.db_save_scan(
                    patient_id, scan_type, fname,
                    result,
                    st.session_state.get("findings_textarea", auto_txt),
                    user["id"],
                )
                if sid:
                    st.success(f"✅ Saved — scan ID: {sid}")
                else:
                    st.error("Save failed. Check database connection.")
            if not can_save:
                st.caption("Select a patient above to enable saving.")

        with c2:
            if st.button(
                "📄 Generate PDF",
                use_container_width=True,
                key="btn_pdf",
            ):
                pat = (
                    db.db_get_patient(patient_id)
                    if patient_id
                    else {
                        "name": "Demo Patient",
                        "patient_id": "DEMO-001",
                        "dob": "—",
                        "gender": "—",
                        "blood_type": "—",
                    }
                )
                pdf = generate_pdf(
                    patient      = pat or {"name": "Unknown", "patient_id": "N/A"},
                    result       = result,
                    findings     = st.session_state.get("findings_textarea", auto_txt),
                    doctor_name  = user.get("full_name") or user["username"],
                    heatmap_arr  = heatmap,
                    original_pil = pil_img,
                )
                ext  = "pdf" if RL_OK else "txt"
                mime = "application/pdf" if RL_OK else "text/plain"
                st.download_button(
                    "⬇️ Download Report",
                    data      = pdf,
                    file_name = f"MedAI_{datetime.now():%Y%m%d_%H%M%S}.{ext}",
                    mime      = mime,
                    use_container_width=True,
                    key="btn_dl_pdf",
                )


# ─────────────────────────────────────────────────────────────────────────────
# ── PATIENTS ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def page_patients() -> None:
    user = st.session_state.user
    st.markdown("## 👥 Patient Management")

    tab_list, tab_add = st.tabs(["📋 Patient List", "➕ Add Patient"])

    with tab_list:
        pats = db.db_get_patients(user["id"])
        if not pats:
            st.info("No patients registered yet. Use the ➕ Add Patient tab.")
        else:
            search = st.text_input("🔍 Search by name or ID", placeholder="Type to filter…")
            if search:
                q = search.lower()
                pats = [
                    p for p in pats
                    if q in p.get("name", "").lower()
                    or q in p.get("patient_id", "").lower()
                ]

            st.markdown(
                '<table class="med-table"><thead><tr>'
                "<th>ID</th><th>Name</th><th>DOB</th>"
                "<th>Gender</th><th>Blood</th></tr></thead><tbody>",
                unsafe_allow_html=True,
            )
            for p in pats:
                st.markdown(
                    f"<tr>"
                    f"<td><code>{p['patient_id']}</code></td>"
                    f"<td><b>{p['name']}</b></td>"
                    f"<td>{str(p.get('dob','—'))[:10]}</td>"
                    f"<td>{p.get('gender','—')}</td>"
                    f"<td>{p.get('blood_type','—')}</td></tr>",
                    unsafe_allow_html=True,
                )
            st.markdown("</tbody></table>", unsafe_allow_html=True)

            # Detail expander
            if pats:
                sel_id = st.selectbox(
                    "View patient details",
                    [p["patient_id"] for p in pats],
                    key="sel_pat_detail",
                )
                p = db.db_get_patient(sel_id)
                if p:
                    scans = db.db_get_scans(sel_id)
                    with st.expander(f"📋 {p['name']} — Full Record", expanded=True):
                        ca, cb = st.columns(2)
                        with ca:
                            for k, lbl in [
                                ("patient_id", "ID"), ("dob", "DOB"),
                                ("gender", "Gender"), ("blood_type", "Blood"),
                            ]:
                                st.markdown(f"**{lbl}:** {p.get(k, '—')}")
                        with cb:
                            for k, lbl in [
                                ("phone", "Phone"), ("email", "Email"),
                                ("notes", "Notes"),
                            ]:
                                st.markdown(f"**{lbl}:** {p.get(k, '—')}")
                        st.markdown(f"**Total Scans:** {len(scans)}")
                        for s in scans[:5]:
                            st.markdown(
                                f'`{str(s.get("created_at",""))[:10]}` '
                                f'**{s.get("scan_type","—")}** — '
                                f'{sev_badge(s.get("severity","—"))}',
                                unsafe_allow_html=True,
                            )

    with tab_add:
        st.markdown("#### Register New Patient")
        with st.form("form_add_patient"):
            ca, cb = st.columns(2)
            with ca:
                name   = st.text_input("Full Name *", placeholder="John Doe")
                dob    = st.date_input(
                    "Date of Birth",
                    value=date(1990, 1, 1),
                    min_value=date(1900, 1, 1),
                )
                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female", "Other", "Prefer not to say"],
                )
            with cb:
                pid   = st.text_input(
                    "Patient ID (auto-generated if blank)",
                    placeholder=generate_patient_id(),
                )
                blood = st.selectbox(
                    "Blood Type",
                    ["A+", "A−", "B+", "B−", "AB+", "AB−", "O+", "O−", "Unknown"],
                )
                phone = st.text_input("Phone", placeholder="+1-555-0000")
            email = st.text_input("Email", placeholder="patient@example.com")
            notes = st.text_area(
                "Clinical Notes",
                height=80,
                placeholder="Allergies, chronic conditions, current medications…",
            )
            add_btn = st.form_submit_button(
                "➕ Register Patient", use_container_width=True, type="primary"
            )

        if add_btn:
            if not name.strip():
                st.error("Patient name is required.")
            else:
                data = {
                    "patient_id": pid.strip() or generate_patient_id(),
                    "name":       name.strip(),
                    "dob":        str(dob),
                    "gender":     gender,
                    "blood_type": blood,
                    "phone":      phone,
                    "email":      email,
                    "notes":      notes,
                }
                ok = db.db_add_patient(data, user["id"])
                if ok:
                    st.success(f"✅ '{data['name']}' registered — ID: {data['patient_id']}")
                else:
                    st.error("Registration failed. Check database connection.")


# ─────────────────────────────────────────────────────────────────────────────
# ── HISTORY ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def page_history() -> None:
    user  = st.session_state.user
    scans = db.db_get_all_scans(user["id"])
    st.markdown("## 📋 Scan History")

    if not scans:
        st.info("No scans recorded yet.")
        if st.button("🔬 Analyze your first image", key="hist_start"):
            _go("analyze")
        return

    f1, f2, f3 = st.columns(3)
    with f1:
        ft = st.selectbox("Scan Type", ["All", "Chest X-Ray", "Brain MRI", "CT Scan"],
                           key="hist_type")
    with f2:
        fs = st.selectbox("Severity",
                           ["All", "Critical", "High", "Moderate", "Low", "Normal"],
                           key="hist_sev")
    with f3:
        fq = st.text_input("Search patient name", placeholder="Filter…", key="hist_q")

    filtered = scans
    if ft != "All":
        filtered = [s for s in filtered if s.get("scan_type") == ft]
    if fs != "All":
        filtered = [s for s in filtered if s.get("severity") == fs]
    if fq:
        ql = fq.lower()
        filtered = [s for s in filtered if ql in s.get("patient_name", "").lower()]

    st.caption(f"Showing {len(filtered)} of {len(scans)} scans")

    st.markdown(
        '<table class="med-table"><thead><tr>'
        "<th>#</th><th>Patient</th><th>Type</th>"
        "<th>Severity</th><th>Preview</th><th>Date</th>"
        "</tr></thead><tbody>",
        unsafe_allow_html=True,
    )
    for i, s in enumerate(filtered[:50], 1):
        sev  = s.get("severity", "—")
        prev = (str(s.get("findings", ""))[:55] + "…") if s.get("findings") else "—"
        dt   = str(s.get("created_at", ""))[:10]
        st.markdown(
            f"<tr><td>{i}</td>"
            f"<td><b>{s.get('patient_name','—')}</b></td>"
            f"<td>{s.get('scan_type','—')}</td>"
            f"<td>{sev_badge(sev)}</td>"
            f'<td style="font-size:.75rem;color:#636e7b">{prev}</td>'
            f"<td>{dt}</td></tr>",
            unsafe_allow_html=True,
        )
    st.markdown("</tbody></table>", unsafe_allow_html=True)

    if filtered:
        df = pd.DataFrame([
            {
                "Patient":   s.get("patient_name", ""),
                "Scan Type": s.get("scan_type", ""),
                "Severity":  s.get("severity", ""),
                "Findings":  str(s.get("findings", ""))[:100],
                "Date":      str(s.get("created_at", ""))[:10],
            }
            for s in filtered
        ])
        st.download_button(
            "⬇️ Export as CSV",
            data      = df.to_csv(index=False),
            file_name = "scan_history.csv",
            mime      = "text/csv",
            key       = "hist_export",
        )


# ─────────────────────────────────────────────────────────────────────────────
# ── REPORTS ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def page_reports() -> None:
    user = st.session_state.user
    st.markdown("## 📄 Generate Reports")

    pats = db.db_get_patients(user["id"])
    if not pats:
        st.info("Add patients and run scans first.")
        return

    pm    = {f"{p['name']} ({p['patient_id']})": p["patient_id"] for p in pats}
    sel_p = st.selectbox("Select Patient", list(pm.keys()), key="rpt_patient")
    pid   = pm[sel_p]
    pat   = db.db_get_patient(pid)
    scans = db.db_get_scans(pid)

    if not scans:
        st.info(f"No scans found for this patient. Analyze an image first.")
        return

    sm    = {
        f"{s.get('scan_type','')} — {s.get('severity','')} "
        f"[{str(s.get('created_at',''))[:10]}]": s
        for s in scans
    }
    sel_s = st.selectbox("Select Scan", list(sm.keys()), key="rpt_scan")
    scan  = sm[sel_s]
    res   = scan.get("result_json", {})
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except (json.JSONDecodeError, TypeError):
            res = {}

    auto_f = scan.get("findings") or build_findings(res)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Patient**")
        for k, lbl in [
            ("name", "Name"), ("patient_id", "ID"),
            ("dob", "DOB"), ("gender", "Gender"),
        ]:
            st.markdown(
                f'<span style="font-size:.82rem"><b>{lbl}:</b> {pat.get(k,"—")}</span><br>',
                unsafe_allow_html=True,
            )
    with cb:
        st.markdown("**Scan**")
        st.markdown(sev_badge(res.get("severity", "—")), unsafe_allow_html=True)
        for k, lbl in [("type", "Type"), ("model", "Model")]:
            st.markdown(
                f'<span style="font-size:.82rem"><b>{lbl}:</b> {res.get(k,"—")}</span><br>',
                unsafe_allow_html=True,
            )

    findings_edit = st.text_area(
        "Edit Findings (optional)", auto_f, height=130, key="rpt_findings"
    )

    if st.button(
        "📄 Generate & Download PDF",
        type="primary",
        use_container_width=True,
        key="rpt_gen",
    ):
        pdf = generate_pdf(
            patient     = pat,
            result      = res,
            findings    = st.session_state.get("rpt_findings", auto_f),
            doctor_name = user.get("full_name") or user["username"],
        )
        ext  = "pdf" if RL_OK else "txt"
        mime = "application/pdf" if RL_OK else "text/plain"
        fn   = f"MedAI_{pat.get('patient_id','')}_{datetime.now():%Y%m%d}.{ext}"
        st.download_button(
            f"⬇️ Download {ext.upper()}",
            data      = pdf,
            file_name = fn,
            mime      = mime,
            use_container_width=True,
            key       = "rpt_dl",
        )
        st.success("✅ Report generated.")


# ─────────────────────────────────────────────────────────────────────────────
# ── SETTINGS ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def page_settings() -> None:
    user = st.session_state.user
    st.markdown("## ⚙️ Settings")

    t1, t2, t3, t4 = st.tabs(
        ["👤 Profile", "🗄️ Database", "🤖 Models", "📥 Data Guide"]
    )

    with t1:
        st.markdown("#### Update Profile")
        with st.form("form_profile"):
            new_name = st.text_input(
                "Full Name", value=user.get("full_name", ""), key="prof_name"
            )
            if st.form_submit_button("Save Changes", type="primary"):
                st.success("✅ Profile updated.")

        st.markdown("---")
        st.markdown("#### Theme")
        mode_lbl = "Dark Mode" if st.session_state.dark_mode else "Light Mode"
        st.write(f"Current: **{mode_lbl}**")
        if st.button("Toggle Dark / Light Mode", key="settings_theme"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    with t2:
        st.markdown("#### Neon PostgreSQL Configuration")
        is_live = db.is_db_available()
        st.markdown(
            f'<div style="padding:12px;'
            f'background:{"#0d2818" if is_live else "#1c1205"};'
            f'border:1px solid {"#22c55e" if is_live else "#a16207"};'
            f'border-radius:10px;margin-bottom:12px">'
            f'<b>{"🟢 Connected — data persists across sessions" if is_live else "🟡 In-Memory mode — data resets on restart"}</b>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.code(
            '# .streamlit/secrets.toml (local) or Streamlit Cloud → Secrets:\n'
            'DATABASE_URL = "postgresql://neondb_owner:PASSWORD'
            '@ep-XXXX.us-east-1.aws.neon.tech/neondb?sslmode=require"'
        )
        st.markdown(
            "Get your connection string from **[neon.tech](https://neon.tech)** "
            "→ Your Project → Connect → Connection String"
        )

    with t3:
        st.markdown("#### Model Status")
        ws = weights_status()
        for scan_t, loaded in ws.items():
            st.markdown(
                f'<div style="background:#161b22;border:1px solid #30363d;'
                f'border-radius:10px;padding:12px 16px;margin-bottom:8px;'
                f'display:flex;justify-content:space-between;align-items:center">'
                f'<b>{scan_t}</b>'
                f'<span style="padding:3px 10px;border-radius:999px;'
                f'font-size:.72rem;font-weight:700;'
                f'background:{"#0d2818" if loaded else "#1c1205"};'
                f'color:{"#4ade80" if loaded else "#f59e0b"}">'
                f'{"✅ Loaded" if loaded else "🔄 Simulated"}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f"**PyTorch:** {'✅ Available' if TORCH_AVAILABLE else '❌ Not installed'}  "
            f"| **ReportLab:** {'✅' if RL_OK else '❌ Not installed'}"
        )
        st.code(
            "# Place trained weights here:\n"
            "weights/brain_mri.pth     ← python train_brain.py\n"
            "weights/chest_xray.pth    ← python train_chest.py"
        )

    with t4:
        st.markdown("#### How to get training data")
        st.markdown("""
**Option A — Download real datasets (recommended):**
```bash
python download_data.py --dataset brain    # ~150 MB
python download_data.py --dataset chest    # ~1.2 GB
python download_data.py --notebook brain   # reference notebook (99% acc)
```
*Requires a free Kaggle account + API token (`~/.kaggle/kaggle.json`)*

**Option B — Generate synthetic data (no download, instant):**
```bash
python generate_synthetic.py --type all        # brain + chest
python generate_synthetic.py --samples 300     # 300 images per class
```

**Train the models:**
```bash
python train_brain.py --epochs 25    # → weights/brain_mri.pth
python train_chest.py --epochs 12    # → weights/chest_xray.pth
```

Place the output `.pth` files in `weights/` — the app detects them automatically.
""")


# ─────────────────────────────────────────────────────────────────────────────
# ── MAIN ROUTER ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if not auth.is_logged_in():
        page_login()
        return

    render_sidebar()

    _PAGES = {
        "dashboard": page_dashboard,
        "analyze":   page_analyze,
        "patients":  page_patients,
        "history":   page_history,
        "reports":   page_reports,
        "settings":  page_settings,
    }
    _PAGES.get(st.session_state.page, page_dashboard)()


if __name__ == "__main__":
    main()