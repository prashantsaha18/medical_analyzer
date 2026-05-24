"""auth.py — Authentication with bcrypt / SHA-256 fallback"""
import hashlib, secrets
import streamlit as st
import database as db

try:
    import bcrypt
    def _hash(pw): return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    def _verify(pw, h):
        try: return bcrypt.checkpw(pw.encode(), h.encode())
        except: return False
except ImportError:
    def _hash(pw):
        s = secrets.token_hex(16)
        return s + ":" + hashlib.sha256((pw+s).encode()).hexdigest()
    def _verify(pw, h):
        try: s, d = h.split(":",1); return hashlib.sha256((pw+s).encode()).hexdigest()==d
        except: return False

def _use_db(): return db.get_conn() is not None

def signup(username, password, full_name="", role="doctor"):
    if len(username) < 3: return False, "Username must be ≥ 3 characters."
    if len(password) < 6: return False, "Password must be ≥ 6 characters."
    hashed = _hash(password)
    if _use_db():
        if db.get_user(username): return False, "Username already taken."
        ok = db.create_user(username, hashed, full_name, role)
        return (True,"Account created!") if ok else (False,"Database error.")
    else:
        if db.mem_get_user(username): return False, "Username already taken."
        db.mem_create_user(username, hashed, full_name, role)
        return True, "Account created!"

def login(username, password):
    user = db.get_user(username) if _use_db() else db.mem_get_user(username)
    if not user: return False, "User not found.", None
    if not _verify(password, user["password"]): return False, "Incorrect password.", None
    return True, "Login successful.", user

def is_logged_in(): return bool(st.session_state.get("user"))
def get_current_user(): return st.session_state.get("user")
def logout():
    for k in ["user","page","result","heatmap","current_img"]:
        st.session_state.pop(k, None)


