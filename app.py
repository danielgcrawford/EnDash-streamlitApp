# app.py

import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from lib import db, auth

st.set_page_config(
    page_title="EnDash - Quick View Dashboard",
    page_icon="🌿",
    layout="wide",
)

# ----- Center title and button text -----
st.markdown(
    """
    <style>
    /* Center all top-level page titles & tighten spacing*/
    h1 {
        text-align: center;
        margin-top: 0rem; /*default is 1*/
        margin-bottom: 0rem;
    }

    /* Center the text inside all Streamlit buttons */
    div.stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }

    /*Tighten vertical space around dividers (st.divider and '---')*/
    hr {
        margin-top: 0rem;
        margin-bottom: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- DB init & admin bootstrap ----------
db.init_db()
auth.ensure_admin()

# ---------- Helpers reused from Upload/Dashboard ----------


def normalize(s: str) -> str:
    """Normalize a column name for matching."""
    s = s.replace("\ufeff", "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)  # "RHT-Temperature" -> "rhttemperature"
    return s


ALIASES = {
    "Time": [
        "time",
        "timestamp",
        "date_time",
        "datetime",
        "recorded at",
        "date.time",
        "logtime",
        "measurement_time",
    ],
    "AirTemp": [
        "airtemp",
        "air_temp",
        "tair",
        "t_air",
        "ambient_temp",
        "air temperature",
        "air temperature (c)",
        "ta_c",
        "rhttemperature",
        "RHT - Temperature",
        "rhttemp",
        "RHT-Temperature",
    ],
    "LeafTemp": [
        "leaftemp",
        "leaf_temp",
        "tleaf",
        "leaf temperature",
        "canopy_temp",
        "tc_leaf",
        "leaf_t (c)",
        "leaf_tc",
    ],
    "RH": [
        "rel_hum",
        "relative_humidity",
        "humidity",
        "rh (%)",
        "rhhumidity",
        "rht_humidity",
        "rh_percent",
    ],
    "PAR": [
        "par",
        "ppfd",
        "photosynthetically active radiation",
        "par_umol",
        "par (umol m-2 s-1)",
        "par_umolm2s",
        "quantum",
        "quantum_sensor",
        "quantumsensor",
        "quantumpar",
    ],
}
CANON_ORDER = ["Time", "AirTemp", "LeafTemp", "RH", "PAR"]


def build_alias_table():
    table = {}
    for canon, aliases in ALIASES.items():
        table[canon] = {normalize(canon), *[normalize(a) for a in aliases]}
    return table


def map_columns(raw_cols, alias_table):
    """Return (mapping {raw -> canon}, missing canonicals, extras)."""
    norm_to_raw = {normalize(c): c for c in raw_cols}
    mapping, used = {}, set()

    # Exact/alias matches
    for canon, norms in alias_table.items():
        for norm, raw in norm_to_raw.items():
            if norm in norms and raw not in used:
                mapping[raw] = canon
                used.add(raw)
                break

    # Fuzzy fallback
    for canon, norms in alias_table.items():
        if canon in mapping.values():
            continue
        for norm, raw in norm_to_raw.items():
            if raw in used:
                continue
            if any(a and a in norm for a in norms):
                mapping[raw] = canon
                used.add(raw)
                break

    missing = [canon for canon in alias_table if canon not in mapping.values()]
    extras = [c for c in raw_cols if c not in mapping]
    return mapping, missing, extras


def load_table_from_bytes(file_bytes: bytes, ext: str):
    """Load CSV or Excel from raw bytes. Returns (df, file_type, encoding_used)."""
    ext = ext.lower()

    if ext in [".xlsx", ".xls"]:
        bio = io.BytesIO(file_bytes)
        df = pd.read_excel(bio)
        return df, "excel", None

    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            bio = io.BytesIO(file_bytes)
            df = pd.read_csv(bio, encoding=enc)
            return df, "csv", enc
        except Exception as e:
            last_err = e

    if last_err is not None:
        raise last_err
    raise ValueError("Could not read file with any of the tried encodings.")


def build_clean_dataframe(df_raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Create clean DataFrame with canonical column names in CANON_ORDER."""
    canon_to_raw = {}
    for raw, canon in mapping.items():
        canon_to_raw.setdefault(canon, raw)

    data = {}
    for canon in CANON_ORDER:
        raw = canon_to_raw.get(canon)
        if raw is None or raw not in df_raw.columns:
            continue

        series = df_raw[raw]
        if canon == "Time":
            series = pd.to_datetime(series, errors="coerce")
        else:
            series = pd.to_numeric(series, errors="coerce")

        data[canon] = series

    if not data:
        return pd.DataFrame()

    df_clean = pd.DataFrame(data)

    if "Time" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["Time"]).sort_values("Time")

    df_clean = df_clean.dropna(axis=0, how="all").drop_duplicates()
    return df_clean


def username_slug(user) -> str:
    base = (
        user.get("username")
        or user.get("email", "").split("@")[0]
        or f"user{user['id']}"
    )
    slug = re.sub(r"[^a-zA-Z0-9]+", "", base).lower()
    return slug or f"user{user['id']}"


def to_celsius(series: pd.Series, orig_is_fahrenheit: bool) -> pd.Series:
    if series is None:
        return None
    return (series - 32.0) * 5.0 / 9.0 if orig_is_fahrenheit else series


def to_display_temp(series_c: pd.Series, unit: str) -> pd.Series:
    if series_c is None:
        return None
    if unit == "F":
        return series_c * 9.0 / 5.0 + 32.0
    return series_c


def diff_to_display(diff_c: pd.Series, unit: str) -> pd.Series:
    if diff_c is None:
        return None
    if unit == "F":
        return diff_c * 9.0 / 5.0
    return diff_c


def pretty_label(col: str, temp_unit: str) -> str:
    temp_symbol = "°F" if temp_unit == "F" else "°C"
    if col == "AirTemp":
        return f"Air temperature ({temp_symbol})"
    if col == "LeafTemp":
        return f"Leaf temperature ({temp_symbol})"
    if col == "RH":
        return "Relative humidity (%)"
    if col == "PAR":
        return "PAR (µmol m⁻² s⁻¹)"
    if col == "VPDleaf":
        return "Leaf VPD (kPa)"
    if col == "VPDair":
        return "Air VPD (kPa)"
    return col


# ---------- Sidebar: user status + custom navigation ----------
with st.sidebar:
    st.title("🌿 EnDash")

    user = auth.current_user()

    # --- User status / logout ---
    if user:
        st.caption(f"Signed in as **{user['username']}**")
        if st.button("Log out"):
            auth.logout()
            st.rerun()
    else:
        st.caption("Not signed in")

    st.divider()
    st.subheader("Navigation")

    # Always show the main page as "Login" instead of "app"
    st.page_link("app.py", label="Login")

    # Only show the rest of the pages once a user is logged in
    if user:
        # Admin page only for admin users
        if user.get("is_admin"):
            st.page_link("pages/0_Admin.py", label="Admin")

        # These are visible to any logged-in user
        st.page_link("pages/1_Upload.py", label="Upload")
        st.page_link("pages/2_Settings.py", label="Settings")
        st.page_link("pages/3_Dashboard.py", label="Dashboard")
        st.page_link("pages/4_Chatbot.py", label="Chatbot")

# ---------- Main content ----------

user = auth.current_user()

st.title("🌿 EnDash")

st.divider()

if not user:
    # ----- Login view -----
    st.subheader("Sign in")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        ok = auth.login(username, password)
        if ok:
            st.success("Logged in.")
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.divider()
    st.caption("Courtesy of the Fisher Lab - IFAS, University of Florida")
    st.stop()

# ----- Logged-in view: Quick View Dashboard -----

# Top-row navigation & actions
if "quick_upload_open" not in st.session_state:
    # Default: show the Quick Upload panel
    st.session_state.quick_upload_open = True

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    if st.button("📂 Manual Upload", use_container_width=True):
        st.switch_page("pages/1_Upload.py")

with col2:
    if st.button("⚙️ Edit Settings", use_container_width=True):
        st.switch_page("pages/2_Settings.py")

with col3:
    if st.button("📊 Full Dashboard", use_container_width=True):
        st.switch_page("pages/3_Dashboard.py")


# ----- Quick Upload panel -----
if st.session_state.quick_upload_open:
    st.markdown("### Quick Upload")
    st.caption(
        "Drop a data file here to generate the Dashboard Summary of your "
        "environmental data. To edit column selections and preview data, use "
        "the Upload page. Edit units and change your desired conditions in the "
        "Settings page. View past reports and further analysis on the Dashboard page."
    )

    quick_file = st.file_uploader(
        "Quick upload (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        key="quick_upload_file",
    )

    # Track the last file we successfully processed so we don't re-process it
    if "last_quick_upload_file_id" not in st.session_state:
        st.session_state.last_quick_upload_file_id = None

    upload_succeeded = False

    if quick_file is not None:
        # Simple ID: (name, size). If this hasn't changed, we already handled it.
        file_id = (quick_file.name, quick_file.size)

        if st.session_state.last_quick_upload_file_id == file_id:
            # Same file as last time and already processed -> do nothing
            pass
        else:
            original_name = quick_file.name
            ext = Path(original_name).suffix or ".csv"
            file_bytes_raw = quick_file.getvalue()

            try:
                # 1) Load raw table
                df_raw, file_type, encoding_used = load_table_from_bytes(
                    file_bytes_raw, ext
                )

                # 2) Automatic column mapping
                alias_table = build_alias_table()
                raw_cols = [str(c) for c in df_raw.columns]
                auto_mapping, _, _ = map_columns(raw_cols, alias_table)

                if not auto_mapping:
                    raise ValueError(
                        "Could not automatically match any columns to "
                        "Time/AirTemp/RH/PAR."
                    )

                # 3) Build cleaned dataframe
                df_clean = build_clean_dataframe(df_raw, auto_mapping)

                required_for_dashboard = ["Time", "AirTemp", "RH"]
                missing_for_dashboard = [
                    c for c in required_for_dashboard if c not in df_clean.columns
                ]
                if missing_for_dashboard:
                    raise ValueError(
                        "Missing required columns for dashboard: "
                        + ", ".join(missing_for_dashboard)
                    )

                # 4) Create stored filename based on data start time
                if "Time" in df_clean.columns and not df_clean["Time"].isna().all():
                    data_start = df_clean["Time"].min()
                    data_start_str = data_start.strftime("%Y%m%dT%H%M")
                else:
                    data_start_str = time.strftime("%Y%m%dT%H%M", time.gmtime())

                uname = username_slug(user)
                stored_filename = f"{uname}_{data_start_str}.csv"

                cleaned_bytes = df_clean.to_csv(index=False).encode("utf-8")
                db.add_file_record(user["id"], stored_filename, cleaned_bytes)

                st.success(
                    f"Quick upload succeeded and cleaned file `{stored_filename}` "
                    "was saved. The quick view dashboard has been updated."
                )

                # Mark this file as processed and close the panel
                st.session_state.last_quick_upload_file_id = file_id
                st.session_state.quick_upload_open = False
                upload_succeeded = True

            except Exception as e:
                st.error(f"Quick upload could not automatically process this file: {e}")
                st.warning(
                    "Use the full Upload page to manually select columns and review "
                    "the mapping for this dataset."
                )
                st.page_link(
                    "pages/1_Upload.py",
                    label="⚠️ Unable to Upload File – Open Upload Page",
                )

    # Trigger a rerun only after a confirmed successful upload
    if upload_succeeded:
        st.rerun()


st.markdown("---")

# ----- Quick View Summary for latest file -----

files = db.list_user_files(user["id"])
if not files:
    st.info(
        "No cleaned files found yet. Use **Quick Upload** above or the "
        "**Upload** page in the sidebar to get started."
    )
    st.stop()

latest = files[0]
st.subheader("Dashboard Summary")
st.caption(
    f"Showing your **latest cleaned file**: `{latest['filename']}` "
    f"(uploaded {latest['uploaded_at']})."
)

file_obj = db.get_file_bytes(latest["id"])
if file_obj is None:
    st.error("Could not load the latest cleaned file from the database.")
    st.stop()

try:
    bio = io.BytesIO(file_obj["bytes"])
    df = pd.read_csv(bio)
except Exception as e:
    st.error(f"Could not read cleaned CSV from Neon: {e}")
    st.stop()

# Ensure Time is datetime & sorted
if "Time" in df.columns:
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time")

# Load per-user settings
settings_row = db.get_or_create_settings(user["id"])
settings = dict(settings_row) if settings_row is not None else {}
temp_unit = settings.get("temp_unit", "F")  # 'F' or 'C'
target_low = float(settings.get("target_low", 65))
target_high = float(settings.get("target_high", 80))

# Core series
air_raw = df["AirTemp"].astype(float) if "AirTemp" in df.columns else None
leaf_raw = df["LeafTemp"].astype(float) if "LeafTemp" in df.columns else None
rh = df["RH"].astype(float) if "RH" in df.columns else None

# Detect original temp units
if air_raw is not None and air_raw.notna().any():
    orig_is_f = air_raw.mean(skipna=True) > 40.0
else:
    orig_is_f = False

# Convert to Celsius for physics / VPD
air_c = to_celsius(air_raw, orig_is_f) if air_raw is not None else None
leaf_c = to_celsius(leaf_raw, orig_is_f) if leaf_raw is not None else None

vpd_air = None
vpd_leaf = None
leaf_air_diff_c = None

if air_c is not None and rh is not None:
    es_air = 0.61121 * np.exp((18.678 - air_c / 234.5) * (air_c / (257.14 + air_c)))
    ea_air = (rh / 100.0) * es_air
    vpd_air = es_air - ea_air
    vpd_air = vpd_air.clip(lower=0)

    if leaf_c is not None:
        es_leaf = 0.61121 * np.exp(
            (18.678 - leaf_c / 234.5) * (leaf_c / (257.14 + leaf_c))
        )
        vpd_leaf = np.maximum(0, es_leaf - ea_air)
        leaf_air_diff_c = leaf_c - air_c

# Convert to display units
air_disp = to_display_temp(air_c, temp_unit) if air_c is not None else None
leaf_disp = to_display_temp(leaf_c, temp_unit) if leaf_c is not None else None
leaf_air_diff_disp = (
    diff_to_display(leaf_air_diff_c, temp_unit) if leaf_air_diff_c is not None else None
)

df_display = df.copy()
if air_disp is not None:
    df_display["AirTemp"] = air_disp
if leaf_disp is not None:
    df_display["LeafTemp"] = leaf_disp
if vpd_leaf is not None:
    df_display["VPDleaf"] = vpd_leaf
elif vpd_air is not None:
    df_display["VPDair"] = vpd_air

# ----- Summary sentence -----
if "Time" in df_display.columns and df_display["Time"].notna().any():
    time_sorted = df_display["Time"].sort_values()
    start_time = time_sorted.iloc[0]
    end_time = time_sorted.iloc[-1]
    st.caption(
        f"Data from **{start_time.strftime('%Y-%m-%d %H:%M:%S')}** "
        f"to **{end_time.strftime('%Y-%m-%d %H:%M:%S')}**."
    )

# ----- Metric widgets -----

metric_cols = st.columns(4)

# 1) Average Air Temperature
if air_disp is not None and air_disp.notna().any():
    air_mean = air_disp.mean(skipna=True)
    if air_mean > target_high:
        delta_text = f"{air_mean - target_high:.1f} above high target"
    elif air_mean < target_low:
        delta_text = f"{target_low - air_mean:.1f} below low target"
    else:
        delta_text = "Within target band"

    metric_cols[0].metric(
        label=f"Avg Air Temp (°{'F' if temp_unit == 'F' else 'C'})",
        value=f"{air_mean:.1f}",
        delta=delta_text,
    )
else:
    metric_cols[0].write("Avg Air Temp: n/a")

# 2) Average Leaf Temperature
if leaf_disp is not None and leaf_disp.notna().any():
    leaf_mean = leaf_disp.mean(skipna=True)
    metric_cols[1].metric(
        label=f"Avg Leaf Temp (°{'F' if temp_unit == 'F' else 'C'})",
        value=f"{leaf_mean:.1f}",
    )
else:
    metric_cols[1].write("Avg Leaf Temp: n/a")

# 3) Average RH
if rh is not None and rh.notna().any():
    rh_mean = rh.mean(skipna=True)
    metric_cols[2].metric(
        label="Avg Relative Humidity (%)",
        value=f"{rh_mean:.0f}",
    )
else:
    metric_cols[2].write("Avg RH: n/a")

# 4) % time within target temperature band
within_pct = None
if air_disp is not None and air_disp.notna().any():
    temp_series = air_disp.dropna()
    if len(temp_series) > 0:
        within_mask = (temp_series >= target_low) & (temp_series <= target_high)
        within_pct = 100.0 * within_mask.mean()

if within_pct is not None:
    metric_cols[3].metric(
        label="Time in target temp band",
        value=f"{within_pct:.0f} %",
    )
else:
    metric_cols[3].write("Time in target band: n/a")

# ----- Issue highlighting -----
if within_pct is not None:
    if within_pct < 50:
        st.error(
            f"Only about **{within_pct:.0f}%** of readings were within your "
            "target temperature band."
        )
    elif within_pct < 80:
        st.warning(
            f"About **{within_pct:.0f}%** of readings were within your target band. "
            
        )
    else:
        st.success(
            f"About **{within_pct:.0f}%** of readings were within your target "
            "temperature band. Conditions were generally close to your targets."
        )

st.markdown("### Key Trends")

# ----- Mini charts -----
if "Time" in df_display.columns:
    df_chart = df_display.set_index("Time")
else:
    df_chart = df_display

chart_cols_temp = [c for c in ["AirTemp", "LeafTemp"] if c in df_chart.columns]
chart_cols_env = [c for c in ["RH"] if c in df_chart.columns]

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if chart_cols_temp:
        st.caption("Air & leaf temperature over time")
        st.line_chart(df_chart[chart_cols_temp])

with chart_col2:
    if chart_cols_env:
        st.caption("Humidity over time")
        st.line_chart(df_chart[chart_cols_env])

st.markdown("---")
st.caption("Courtesy of the Fisher Lab - IFAS, University of Florida")
