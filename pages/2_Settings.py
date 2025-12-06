# pages/2_Settings.py
import streamlit as st
from lib import auth, db

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="centered")
auth.require_login()
user = auth.current_user()
auth.render_sidebar()

st.title("⚙️ Settings")

st.markdown(
    "These temperature preferences are saved **per account** and will be used "
    "for calculations and graphing on the dashboard."
)

# ---- Load (or create) per-user settings ----
row = db.get_or_create_settings(user["id"])
# Convert to a plain dict so we can use .get() safely
row = dict(row) if row is not None else {}

# Fallback defaults if the DB is missing any fields (e.g. after schema change)
temp_unit = row.get("temp_unit", "F")          # 'F' or 'C'
target_low = float(row.get("target_low", 65))  # default 65°F
target_high = float(row.get("target_high", 80))  # default 80°F

with st.form("settings_form"):
    # ----- Temperature unit selection -----
    unit_display_options = ["Fahrenheit (°F)", "Celsius (°C)"]
    unit_index = 0 if temp_unit == "F" else 1
    unit_choice = st.selectbox("Temperature unit", unit_display_options, index=unit_index)

    # Map display label back to a simple code for storage
    selected_unit = "F" if "Fahrenheit" in unit_choice else "C"

    # ----- Target temperatures -----
    col1, col2 = st.columns(2)
    with col1:
        low_input = st.number_input(
            "Target low temperature",
            value=target_low,
            step=0.5,
            format="%.2f",
            help="The lower threshold you care about for your environment."
        )
    with col2:
        high_input = st.number_input(
            "Target high temperature",
            value=target_high,
            step=0.5,
            format="%.2f",
            help="The upper threshold you care about for your environment."
        )

    save_btn = st.form_submit_button("Save settings")

# ---- Save logic ----
if save_btn:
    if high_input <= low_input:
        st.error("Target high temperature must be greater than target low temperature.")
    else:
        db.update_settings(
            user["id"],
            selected_unit,
            float(low_input),
            float(high_input),
        )
        st.success("Your personal settings have been saved and will be used on the dashboard.")
