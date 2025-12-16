# lib/auth.py
import streamlit as st
from passlib.hash import bcrypt_sha256
from . import db

def hash_password(plaintext: str) -> str:
    """Hash with bcrypt+sha256 to avoid bcrypt's 72 byte limit."""
    return bcrypt_sha256.hash(plaintext)

def verify_password(plaintext: str, password_hash: str) -> bool:
    return bcrypt_sha256.verify(plaintext, password_hash)

def ensure_admin():
    """Create an admin if missing, using secrets or defaults."""
    username = st.secrets.get("ADMIN_USERNAME", "admin")
    password = st.secrets.get("ADMIN_PASSWORD", "change-this-now")
    existing = db.get_user_by_username(username)
    if existing is None:
        h = hash_password(password)
        db.add_user(username, h, is_admin=1)

def login(username: str, password: str) -> bool:
    row = db.get_user_by_username(username)
    if not row:
        return False
    if not verify_password(password, row["password_hash"]):
        return False
    st.session_state["user"] = {
        "id": row["id"],
        "username": row["username"],
        "is_admin": bool(row["is_admin"]),
    }
    return True

def logout():
    st.session_state.pop("user", None)

def current_user():
    return st.session_state.get("user")

def require_login():
    """Call at the top of each page to enforce auth."""
    if "user" not in st.session_state:
        st.warning("Please log in first.")
        st.stop()

def render_sidebar():
    """Render the shared sidebar with nav on every page."""
    with st.sidebar:
        st.title("🌿 EnDash")

        user = current_user()

        # --- User status / logout ---
        if user:
            st.caption(f"Signed in as **{user['username']}**")
            if st.button("Log out"):
                logout()
                st.rerun()
        else:
            st.caption("Not signed in")

        st.divider()
        st.subheader("Navigation")

        # Main page shows as 'Login' instead of 'app'
        st.page_link("app.py", label="Home")

        if user:
            # Admin tab only for admins
            if user.get("is_admin"):
                st.page_link("pages/0_Admin.py", label="Admin")

            # Other pages for any logged-in user
            st.page_link("pages/1_Upload.py", label="Upload Files")
            st.page_link("pages/2_Settings.py", label="Settings & Setpoints")
            st.page_link("pages/4_Chatbot.py", label="Chatbot")


#Update Admin Password - 11/18/25
def change_password(user_id: int, current_plaintext: str, new_plaintext: str) -> bool:
    """Verify current password, then set a new one. Returns True if changed."""
    row = db.get_user_by_id(user_id)
    if not row:
        return False
    if not verify_password(current_plaintext, row["password_hash"]):
        return False
    new_hash = hash_password(new_plaintext)
    db.update_user_password(user_id, new_hash)
    return True
