# 2_Settings.py

import streamlit as st
from lib import auth, db

# ---------------------------------------------------------
# Page setup & auth
# ---------------------------------------------------------
st.set_page_config(
    page_title="Settings & Setpoints",
    page_icon="⚙️",
    layout="centered",
)
auth.require_login()
user = auth.current_user()
auth.render_sidebar()

st.title("⚙️ Settings & Setpoints")

st.markdown(
    """
These preferences are saved **per account** and will be used for unit
conversions, calculations, and graphing on the dashboard.
"""
)

# ---------------------------------------------------------
# DEFAULTS – edit these to change app-wide default behavior
# ---------------------------------------------------------
# UNITS
DEFAULT_ORIG_TEMP_UNIT = "C"      # 'C' or 'F'  (data file)
DEFAULT_ORIG_LIGHT_UNIT = "PPFD"  # 'PPFD', 'LUX', 'KLUX', 'FC', 'W_M2'
DEFAULT_DASHBOARD_TEMP_UNIT = "F" # 'C' or 'F'  (dashboard display)

# TEMPERATURE TARGETS
DEFAULT_TARGET_TEMP_LOW = 65.0   # e.g. °F
DEFAULT_TARGET_TEMP_HIGH = 80.0

# RH TARGETS (%)
DEFAULT_TARGET_RH_LOW = 70.0
DEFAULT_TARGET_RH_HIGH = 95.0

# LIGHT / DLI TARGETS
DEFAULT_TARGET_PPFD = 150.0   # µmol m⁻² s⁻¹
DEFAULT_TARGET_DLI = 8.0      # mol m⁻² d⁻¹

# VPD TARGETS (kPa)
DEFAULT_TARGET_VPD_LOW = 0.2
DEFAULT_TARGET_VPD_HIGH = 0.8

# IRRIGATION EVENT DETECTION 
DEFAULT_IRRIGATION_TRIGGER = 1.0    #irrigation ON when value >= this
DEFAULT_IRRIGATION_MIN_INTERVAL_MIN = 7.0   #minimum minutes between counted events

# LEAF WETNESS IRRIGATION DETECTION
DEFAULT_LEAF_WETNESS_UNIT = "Percent"
DEFAULT_IRRIGATION_SENSITIVITY_PCT = 5.0    #when increase in LW >= this, irrigation is counted ON
DEFAULT_LEAF_WETNESS_MIN_INTERVAL_MIN = 7.0 #min minutes between counted irrigation events

# ---------------------------------------------------------
# Load per-user settings from DB
# ---------------------------------------------------------
row = db.get_or_create_settings(user["id"])
settings = dict(row) if row is not None else {}

orig_temp_unit = settings.get("orig_temp_unit", DEFAULT_ORIG_TEMP_UNIT)
orig_light_unit = settings.get("orig_light_unit", DEFAULT_ORIG_LIGHT_UNIT)
dashboard_temp_unit = settings.get("temp_unit", DEFAULT_DASHBOARD_TEMP_UNIT)

target_temp_low = float(settings.get("target_low", DEFAULT_TARGET_TEMP_LOW))
target_temp_high = float(settings.get("target_high", DEFAULT_TARGET_TEMP_HIGH))

target_rh_low = float(settings.get("target_rh_low", DEFAULT_TARGET_RH_LOW))
target_rh_high = float(settings.get("target_rh_high", DEFAULT_TARGET_RH_HIGH))

target_ppfd = float(settings.get("target_ppfd", DEFAULT_TARGET_PPFD))
target_dli = float(settings.get("target_dli", DEFAULT_TARGET_DLI))

target_vpd_low = float(settings.get("target_vpd_low", DEFAULT_TARGET_VPD_LOW))
target_vpd_high = float(settings.get("target_vpd_high", DEFAULT_TARGET_VPD_HIGH))

irrigation_trigger = float(settings.get("irrigation_trigger", DEFAULT_IRRIGATION_TRIGGER))
irrigation_min_interval_min = float(settings.get("irrigation_min_interval_min", DEFAULT_IRRIGATION_MIN_INTERVAL_MIN))

leaf_wetness_unit = settings.get("leaf_wetness_unit", DEFAULT_LEAF_WETNESS_UNIT)
irrigation_sensitivity_pct = float(settings.get("irrigation_sensitivity_pct", DEFAULT_IRRIGATION_SENSITIVITY_PCT))
leaf_wetness_min_interval_min = float(settings.get("leaf_wetness_min_interval_min", DEFAULT_LEAF_WETNESS_MIN_INTERVAL_MIN))

# ---------------------------------------------------------
# UI helpers for unit labels <-> codes
# ---------------------------------------------------------
TEMP_UNIT_OPTIONS = {
    "Celsius (°C)": "C",
    "Fahrenheit (°F)": "F",
}

LIGHT_UNIT_OPTIONS = {
    "PPFD (µmol m⁻² s⁻¹)": "PPFD",
    "Lux": "LUX",
    "Kilolux (klux)": "KLUX",
    "Footcandles (fc)": "FC",
    "W m⁻² (broadband irradiance)": "W_M2",
}


def temp_unit_display_index(current_code: str) -> int:
    """Return index into TEMP_UNIT_OPTIONS.keys() for a given code."""
    labels = list(TEMP_UNIT_OPTIONS.keys())
    for i, lbl in enumerate(labels):
        if TEMP_UNIT_OPTIONS[lbl] == current_code:
            return i
    return 0


def light_unit_display_index(current_code: str) -> int:
    """Return index into LIGHT_UNIT_OPTIONS.keys() for a given code."""
    labels = list(LIGHT_UNIT_OPTIONS.keys())
    for i, lbl in enumerate(labels):
        if LIGHT_UNIT_OPTIONS[lbl] == current_code:
            return i
    return 0


# ---------------------------------------------------------
# Settings form
# ---------------------------------------------------------
with st.form("settings_form"):
    # =====================================================
    # UNITS SECTION
    # =====================================================
    st.subheader("Units")
    st.caption(
        "Tell EnDash what units your **data file uses** and what units you "
        "want to **see** on the dashboard. These preferences are saved per user."
    )

    # ------- Original Data File Units -------
    st.markdown("#### Original data file units")

    col_orig_temp, col_orig_light, col_orig_lw = st.columns(3)

    with col_orig_temp:
        orig_temp_labels = list(TEMP_UNIT_OPTIONS.keys())
        orig_temp_idx = temp_unit_display_index(orig_temp_unit)
        orig_temp_choice = st.selectbox(
            "Temperature",
            orig_temp_labels,
            index=orig_temp_idx,
            help="Units used by the temperature column(s) in your uploaded file.",
        )
        selected_orig_temp_unit = TEMP_UNIT_OPTIONS[orig_temp_choice]

    with col_orig_light:
        light_labels = list(LIGHT_UNIT_OPTIONS.keys())
        light_idx = light_unit_display_index(orig_light_unit)
        light_choice = st.selectbox(
            "Light",
            light_labels,
            index=light_idx,
            help="Units used by the light / PAR column in your uploaded file.",
        )
        selected_orig_light_unit = LIGHT_UNIT_OPTIONS[light_choice]

    with col_orig_lw:
        lw_options = ["Percent", "Volts", "milliVolts"]

        if leaf_wetness_unit in (None, "", "%") or leaf_wetness_unit not in lw_options:
            leaf_wetness_unit = "Percent"

        leaf_wetness_unit_input = st.selectbox(
            "Leaf Wetness",
            options=lw_options,
            index=lw_options.index(leaf_wetness_unit),
            help="Units used by the Leaf Wetness column in your uploaded file.",
        )

    # ------- Desired Dashboard Units -------
    st.markdown("#### Desired dashboard units")

    dash_temp_labels = list(TEMP_UNIT_OPTIONS.keys())
    dash_temp_idx = temp_unit_display_index(dashboard_temp_unit)
    dash_temp_choice = st.selectbox(
        "Temperature on dashboard",
        dash_temp_labels,
        index=dash_temp_idx,
        help="Units used when displaying temperatures on the dashboard and PDF.",
    )
    selected_dashboard_temp_unit = TEMP_UNIT_OPTIONS[dash_temp_choice]

    st.markdown("---")

    # =====================================================
    # TARGET SETPOINTS SECTION
    # =====================================================
    st.subheader("Target setpoints")
    st.caption(
        "These targets are used to summarize and highlight your environmental "
        "conditions on the dashboard. You can tune them to match your crop "
        "and propagation stage."
    )

    # ------- Temperature setpoints -------
    st.markdown("##### Temperature targets")

    col_t_low, col_t_high = st.columns(2)
    with col_t_low:
        temp_low_input = st.number_input(
            "Target low temperature",
            value=target_temp_low,
            step=0.5,
            format="%.2f",
            help="Lower bound of your desired temperature band.",
        )
    with col_t_high:
        temp_high_input = st.number_input(
            "Target high temperature",
            value=target_temp_high,
            step=0.5,
            format="%.2f",
            help="Upper bound of your desired temperature band.",
        )

    # ------- RH setpoints -------
    st.markdown("##### Relative humidity targets")

    col_rh_low, col_rh_high = st.columns(2)
    with col_rh_low:
        rh_low_input = st.number_input(
            "Target low RH (%)",
            value=target_rh_low,
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            help="Lower bound of your desired relative humidity band.",
        )
    with col_rh_high:
        rh_high_input = st.number_input(
            "Target high RH (%)",
            value=target_rh_high,
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            help="Upper bound of your desired relative humidity band.",
        )

    # ------- Light & DLI targets -------
    st.markdown("##### Light & DLI targets")

    col_ppfd, col_dli = st.columns(2)
    with col_ppfd:
        ppfd_input = st.number_input(
            "Target PPFD (µmol m⁻² s⁻¹)",
            value=target_ppfd,
            min_value=0.0,
            step=10.0,
            format="%.1f",
            help="Target instantaneous PAR intensity.",
        )
    with col_dli:
        dli_input = st.number_input(
            "Target Daily Light Integral (mol m⁻² d⁻¹)",
            value=target_dli,
            min_value=0.0,
            step=0.5,
            format="%.2f",
            help="Daily light integral target for the crop.",
        )

    # ------- VPD setpoints -------
    st.markdown("##### VPD targets")

    col_vpd_low, col_vpd_high = st.columns(2)
    with col_vpd_low:
        vpd_low_input = st.number_input(
            "Target low VPD (kPa)",
            value=target_vpd_low,
            min_value=0.0,
            step=0.05,
            format="%.2f",
            help="Lower bound of your desired VPD band.",
        )
    with col_vpd_high:
        vpd_high_input = st.number_input(
            "Target high VPD (kPa)",
            value=target_vpd_high,
            min_value=0.0,
            step=0.05,
            format="%.2f",
            help="Upper bound of your desired VPD band.",
        )
    
    # =====================================================
    # IRRIGATION SECTION
    # =====================================================
    st.markdown("---")
    st.subheader("Irrigation")
    st.caption(
        "These settings control how EnDash converts irrigation signals into ON/OFF states "
        "and how it calculates water applied per day."
    )

    col_it, col_gap = st.columns(2)

    with col_it:
        irrigation_trigger_input = st.number_input(
            "Irrigation Trigger (ON when value ≥ trigger)",
            value=float(irrigation_trigger),
            min_value=0.1,      # per your requirement: default and minimum >= 1
            step=0.1,
            format="%.1f",
            help="Your irrigation column is numeric: 0 = off, values ≥ trigger = on.",
        )

    with col_gap:
        irrigation_min_interval_input = st.number_input(
            "Minimum Time Between Irrigation Events (minutes)",
            value=float(irrigation_min_interval_min),
            min_value=0.0,
            step=1.0,
            format="%.0f",
            help=(
                "Prevents counting the same irrigation run multiple times when it spans "
                "multiple logging steps (e.g., 12:30 and 12:31)."
            ),
        )

    st.subheader("Leaf Wetness")
    st.caption(
        "These settings control how EnDash converts Leaf Wetness into irrigation signals."
    )

    col_is, col_lw_gap = st.columns(2)

    with col_lw_gap:
        leaf_wetness_min_interval_input = st.number_input(
            "Minimum Time Between Irrigation Events (minutes)",
            value=float(leaf_wetness_min_interval_min),
            min_value=0.0,
            step=1.0,
            format="%.0f",
            help=(
                "Prevents counting multiple Leaf Wetness spikes as separate irrigation events. "
                "An event will only be counted if enough time has passed since the last event."
            ),
        )

    with col_is:
        irrigation_sensitivity_pct_input = st.number_input(
            "Irrigation Sensitivity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(irrigation_sensitivity_pct),
            step=0.1,
            help="Minimum percent rise in Leaf Wetness between consecutive readings to count an irrigation event.",
        )

    save_btn = st.form_submit_button("💾 Save settings")

# ---------------------------------------------------------
# Save logic
# ---------------------------------------------------------
if save_btn:
    errors = []

    if temp_high_input <= temp_low_input:
        errors.append("Target **high** temperature must be greater than target **low** temperature.")
    if rh_high_input <= rh_low_input:
        errors.append("Target **high RH** must be greater than target **low RH**.")
    if vpd_high_input <= vpd_low_input:
        errors.append("Target **high VPD** must be greater than target **low VPD**.")

    if errors:
        for err in errors:
            st.error(err)
    else:
        db.update_settings(
            user["id"],
            orig_temp_unit=selected_orig_temp_unit,
            orig_light_unit=selected_orig_light_unit,
            temp_unit=selected_dashboard_temp_unit,
            target_low=float(temp_low_input),
            target_high=float(temp_high_input),
            target_rh_low=float(rh_low_input),
            target_rh_high=float(rh_high_input),
            target_ppfd=float(ppfd_input),
            target_dli=float(dli_input),
            target_vpd_low=float(vpd_low_input),
            target_vpd_high=float(vpd_high_input),
            irrigation_trigger=float(irrigation_trigger_input),
            irrigation_min_interval_min=float(irrigation_min_interval_input),
            leaf_wetness_unit=leaf_wetness_unit_input,
            irrigation_sensitivity_pct=float(irrigation_sensitivity_pct_input),
            leaf_wetness_min_interval_min=float(leaf_wetness_min_interval_input),
        )

        st.success("Your personal units and setpoints have been saved and will be used on the dashboard.")
