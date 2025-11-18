# pages/0_Admin.py
#New Admin Page creaated - 11/18/25
import streamlit as st
import sqlite3
from lib import auth, db

st.set_page_config(page_title="Admin", page_icon="🔒", layout="centered")
auth.require_login()
user = auth.current_user()

if not user["is_admin"]:
    st.error("Admins only.")
    st.stop()

st.title("🔒 Admin")

# ---------- One-time Admin Password Change ----------
finalized = (db.get_meta("admin_pw_finalized") == "1")

if not finalized:
    st.subheader("Set Admin Password (one-time)")
    with st.form("change_admin_pw"):
        current = st.text_input("Current admin password", type="password")
        new = st.text_input("New admin password", type="password")
        confirm = st.text_input("Confirm new password", type="password")
        submit_pw = st.form_submit_button("Update admin password")

    if submit_pw:
        if new != confirm:
            st.error("New passwords do not match.")
        elif len(new) < 8:
            st.error("Use at least 8 characters.")
        else:
            ok = auth.change_password(user["id"], current, new)
            if ok:
                db.set_meta("admin_pw_finalized", "1")   # hide this form forever
                st.success("Admin password updated. This section is now finalized.")
                st.experimental_rerun()
            else:
                st.error("Current password is incorrect.")
else:
    st.info("Admin password has been finalized.")

st.divider()

# ---------- Create New User Accounts ----------
st.subheader("Create New User")
with st.form("add_user"):
    uname = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    make_admin = st.checkbox("Grant admin privileges?", value=False)
    submit_user = st.form_submit_button("Create user")

if submit_user:
    if not uname or not pw:
        st.error("Username and password are required.")
    elif len(pw) < 8:
        st.error("Password must be at least 8 characters.")
    else:
        try:
            pwd_hash = auth.hash_password(pw)
            new_id = db.add_user(uname, pwd_hash, 1 if make_admin else 0)
            st.success(f"User '{uname}' created (id={new_id}).")
        except sqlite3.IntegrityError:
            st.error("Username already exists.")
