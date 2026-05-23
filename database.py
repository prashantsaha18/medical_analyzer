"""database.py — Neon PostgreSQL + in-memory fallback for MedAI"""
import os, json
from datetime import datetime
from typing import Optional

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PG_OK = True
except ImportError:
    PG_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────
def get_conn():
    if not PG_OK:
        return None
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        try:
            import streamlit as st
            dsn = st.secrets.get("DATABASE_URL", "")
        except Exception:
            pass
    if not dsn:
        return None
    try:
        return psycopg2.connect(dsn, sslmode="require", connect_timeout=8)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id         SERIAL PRIMARY KEY,
    username   TEXT UNIQUE NOT NULL,
    password   TEXT NOT NULL,
    full_name  TEXT DEFAULT '',
    role       TEXT DEFAULT 'doctor',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS patients (
    id          SERIAL PRIMARY KEY,
    patient_id  TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    dob         TEXT,
    gender      TEXT,
    blood_type  TEXT,
    phone       TEXT,
    email       TEXT,
    notes       TEXT,
    created_by  INTEGER,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS scans (
    id          SERIAL PRIMARY KEY,
    patient_id  TEXT,
    scan_type   TEXT,
    filename    TEXT,
    result_json JSONB,
    severity    TEXT,
    findings    TEXT,
    doctor_id   INTEGER,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS reports (
    id          SERIAL PRIMARY KEY,
    scan_id     INTEGER,
    report_text TEXT,
    signed_by   INTEGER,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
"""

def init_db() -> bool:
    conn = get_conn()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute(_SCHEMA)
        conn.commit(); cur.close(); conn.close()
        return True
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# USERS
# ─────────────────────────────────────────────────────────────────────────────
def create_user(username, hashed_pw, full_name="", role="doctor") -> bool:
    conn = get_conn()
    if not conn: return False
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username,password,full_name,role) "
                    "VALUES (%s,%s,%s,%s)", (username, hashed_pw, full_name, role))
        conn.commit(); cur.close(); conn.close(); return True
    except Exception: return False

def get_user(username: str) -> Optional[dict]:
    conn = get_conn()
    if not conn: return None
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        row = cur.fetchone(); cur.close(); conn.close()
        return dict(row) if row else None
    except Exception: return None

# ─────────────────────────────────────────────────────────────────────────────
# PATIENTS
# ─────────────────────────────────────────────────────────────────────────────
def add_patient(data: dict, created_by: int) -> bool:
    conn = get_conn()
    if not conn: return False
    try:
        cur = conn.cursor()
        cur.execute("""INSERT INTO patients
            (patient_id,name,dob,gender,blood_type,phone,email,notes,created_by)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (patient_id) DO UPDATE SET name=EXCLUDED.name""",
            (data["patient_id"],data["name"],data.get("dob"),data.get("gender"),
             data.get("blood_type"),data.get("phone"),data.get("email"),
             data.get("notes",""),created_by))
        conn.commit(); cur.close(); conn.close(); return True
    except Exception as e: print(f"add_patient: {e}"); return False

def get_patients(uid: int) -> list:
    conn = get_conn()
    if not conn: return []
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM patients WHERE created_by=%s ORDER BY created_at DESC", (uid,))
        rows = [dict(r) for r in cur.fetchall()]; cur.close(); conn.close(); return rows
    except Exception: return []

def get_patient(pid: str) -> Optional[dict]:
    conn = get_conn()
    if not conn: return None
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM patients WHERE patient_id=%s", (pid,))
        row = cur.fetchone(); cur.close(); conn.close()
        return dict(row) if row else None
    except Exception: return None

# ─────────────────────────────────────────────────────────────────────────────
# SCANS
# ─────────────────────────────────────────────────────────────────────────────
def save_scan(patient_id, scan_type, filename, result, findings, doctor_id) -> Optional[int]:
    conn = get_conn()
    if not conn: return None
    try:
        cur = conn.cursor()
        cur.execute("""INSERT INTO scans
            (patient_id,scan_type,filename,result_json,severity,findings,doctor_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (patient_id,scan_type,filename,json.dumps(result),
             result.get("severity",""),findings,doctor_id))
        sid = cur.fetchone()[0]; conn.commit(); cur.close(); conn.close(); return sid
    except Exception as e: print(f"save_scan: {e}"); return None

def get_scans(patient_id: str) -> list:
    conn = get_conn()
    if not conn: return []
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM scans WHERE patient_id=%s ORDER BY created_at DESC",(patient_id,))
        rows = [dict(r) for r in cur.fetchall()]; cur.close(); conn.close(); return rows
    except Exception: return []

def get_all_scans(doctor_id: int, limit=50) -> list:
    conn = get_conn()
    if not conn: return []
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""SELECT s.*, p.name as patient_name FROM scans s
            LEFT JOIN patients p ON s.patient_id=p.patient_id
            WHERE s.doctor_id=%s ORDER BY s.created_at DESC LIMIT %s""",
            (doctor_id, limit))
        rows = [dict(r) for r in cur.fetchall()]; cur.close(); conn.close(); return rows
    except Exception: return []

def get_dashboard_stats(uid: int) -> dict:
    conn = get_conn()
    if not conn: return {"patients":0,"scans":0,"critical":0,"reports":0}
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM patients WHERE created_by=%s",(uid,))
        p = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM scans WHERE doctor_id=%s",(uid,))
        s = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM scans WHERE doctor_id=%s AND severity IN ('Critical','High')",(uid,))
        c = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM reports r JOIN scans sc ON r.scan_id=sc.id WHERE sc.doctor_id=%s",(uid,))
        r = cur.fetchone()[0]
        cur.close(); conn.close()
        return {"patients":p,"scans":s,"critical":c,"reports":r}
    except Exception: return {"patients":0,"scans":0,"critical":0,"reports":0}

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
_M: dict = {"users":{},"patients":{},"scans":[],"uid":1,"sid":1}

def mem_create_user(username, pw, full_name="", role="doctor"):
    _M["users"][username] = {"id":_M["uid"],"username":username,"password":pw,
        "full_name":full_name,"role":role,"created_at":datetime.now().isoformat()}
    _M["uid"] += 1

def mem_get_user(username): return _M["users"].get(username)

def mem_add_patient(data, created_by):
    _M["patients"][data["patient_id"]] = {**data,"created_by":created_by,
        "created_at":datetime.now().isoformat()}

def mem_get_patients(uid):
    return [p for p in _M["patients"].values() if p.get("created_by")==uid]

def mem_get_patient(pid): return _M["patients"].get(pid)

def mem_save_scan(patient_id, scan_type, filename, result, findings, doctor_id):
    sid = _M["sid"]
    _M["scans"].append({"id":sid,"patient_id":patient_id,"scan_type":scan_type,
        "filename":filename,"result_json":result,"severity":result.get("severity",""),
        "findings":findings,"doctor_id":doctor_id,"created_at":datetime.now().isoformat()})
    _M["sid"] += 1; return sid

def mem_get_scans(patient_id):
    return [s for s in _M["scans"] if s["patient_id"]==patient_id]

def mem_get_all_scans(uid, limit=50):
    scans = [s for s in _M["scans"] if s["doctor_id"]==uid]
    for s in scans:
        s["patient_name"] = _M["patients"].get(s["patient_id"],{}).get("name","Unknown")
    return sorted(scans, key=lambda x:x["created_at"], reverse=True)[:limit]

def mem_get_stats(uid):
    pts   = len([p for p in _M["patients"].values() if p.get("created_by")==uid])
    scans = [s for s in _M["scans"] if s["doctor_id"]==uid]
    crit  = len([s for s in scans if s.get("severity") in ("Critical","High")])
    return {"patients":pts,"scans":len(scans),"critical":crit,"reports":0}
