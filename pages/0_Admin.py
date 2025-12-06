# pages/0_Admin.py
#New Admin Page creaated - 11/18/25
import streamlit as st
from psycopg2 import errors
from lib import auth, db

st.set_page_config(page_title="Admin", page_icon="🔒", layout="centered")
auth.require_login()
user = auth.current_user()

auth.render_sidebar()

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
        except errors.UniqueViolation:
            st.error("Username already exists.")
        except Exception as e:
            st.error("Error creating user. Please try again.")

st.divider()
st.subheader("Existing Users")

rows = db.list_users()
if rows:
    st.table(
        {
            "ID": [r["id"] for r in rows],
            "Username": [r["username"] for r in rows],
            "Admin?": [bool(r["is_admin"]) for r in rows],
            "Created at": [r["created_at"] for r in rows],
        }
    )
else:
    st.info("No users found.")


st.divider()
st.subheader("Existing users")

users = db.list_users()

if not users:
    st.info("No users found.")
else:
    for row in users:
        user_id = row["id"]
        username = row["username"]
        is_admin = bool(row["is_admin"])
        created_at = row["created_at"]

        with st.expander(
            f"{username}  |  "
            f"{'Admin' if is_admin else 'User'}  |  "
            f"Created: {created_at}",
            expanded=False,
        ):
            new_pw = st.text_input(
                "New password",
                type="password",
                key=f"new_pw_{user_id}",
            )
            confirm_pw = st.text_input(
                "Confirm new password",
                type="password",
                key=f"confirm_pw_{user_id}",
            )

            if st.button("Update password", key=f"update_pw_{user_id}"):
                if not new_pw or not confirm_pw:
                    st.error("Both password fields are required.")
                elif new_pw != confirm_pw:
                    st.error("Passwords do not match.")
                elif len(new_pw) < 8:
                    st.error("Password must be at least 8 characters.")
                else:
                    new_hash = auth.hash_password(new_pw)
                    db.update_user_password(user_id, new_hash)
                    st.success(f"Password updated for {username}.")
