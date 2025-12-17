import io
import re
import time
from io import BytesIO
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

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
auth.render_sidebar()

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


def format_timedelta(td) -> str:
    """Pretty-print a pandas Timedelta."""
    if td is None:
        return "unknown interval"
    seconds = int(td.total_seconds())
    if seconds < 60:
        return f"{seconds} seconds"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minutes"
    hours = minutes // 60
    minutes = minutes % 60
    if hours < 24:
        if minutes == 0:
            return f"{hours} hours"
        return f"{hours} hours {minutes} minutes"
    days = hours // 24
    hours = hours % 24
    return f"{days} days {hours} hours"


def apply_time_axis_formatting(ax, fig, x_values):
    """Apply reasonable tick spacing & formatting for a datetime x-axis."""
    time_min = x_values.min()
    time_max = x_values.max()
    total_seconds = (time_max - time_min).total_seconds()

    if total_seconds <= 6 * 3600:  # ≤ 6 hours
        locator = mdates.MinuteLocator(interval=10)
        formatter = mdates.DateFormatter("%H:%M")
    elif total_seconds <= 24 * 3600:  # ≤ 1 day
        locator = mdates.HourLocator(interval=2)
        formatter = mdates.DateFormatter("%H:%M")
    elif total_seconds <= 7 * 24 * 3600:  # ≤ 1 week
        locator = mdates.DayLocator(interval=1)
        formatter = mdates.DateFormatter("%m-%d")
    else:
        locator = mdates.AutoDateLocator()
        formatter = mdates.AutoDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=30, ha="right")

#Naming
def pretty_label(col: str, temp_unit: str) -> str:
    """Human-readable labels with units (write out words instead of abbreviations)."""
    temp_symbol = "°F" if temp_unit == "F" else "°C"
    if col == "AirTemp":
        return f"Air Temperature ({temp_symbol})"
    if col == "LeafTemp":
        return f"Leaf Temperature ({temp_symbol})"
    if col == "RH":
        return "Relative Humidity (%)"
    if col == "PAR":
        return "Light Intensity (PPFD - µmol m⁻² s⁻¹)"
    if col == "VPDleaf":
        return "Leaf Vapor Pressure Deficit (kPa)"
    if col == "VPDair":
        return "Air Vapor Pressure Deficit (kPa)"
    return col


def compute_daily_dli(df_light: pd.DataFrame) -> pd.Series | None:
    """
    Compute Daily Light Integral (DLI) from PPFD measurements.

    DLI (mol m⁻² d⁻¹) = Σ (PPFD_i * Δt) / 1,000,000

    Notes:
      - Uses median logging interval for Δt.
      - Only computes DLI for "full days" (>=20h span and >=80% expected samples).
    """
    if "Time" not in df_light.columns or "PAR" not in df_light.columns:
        return None

    df_light = df_light.dropna(subset=["Time", "PAR"]).copy()
    if df_light.empty or df_light["Time"].nunique() < 2:
        return None

    df_light = df_light.sort_values("Time")
    dt_series = df_light["Time"].diff().dt.total_seconds().dropna()
    if dt_series.empty:
        return None

    median_dt = float(dt_series.median())
    if median_dt <= 0:
        return None

    seconds_per_day = 24.0 * 3600.0
    df_light["Date"] = df_light["Time"].dt.date
    daily_dlis = {}

    for date, group in df_light.groupby("Date"):
        n = len(group)
        span_sec = (group["Time"].iloc[-1] - group["Time"].iloc[0]).total_seconds()
        expected_n = seconds_per_day / median_dt

        if span_sec < 20.0 * 3600.0:
            continue
        if n < 0.8 * expected_n:
            continue

        dli_umol = group["PAR"].sum() * median_dt
        dli_mol = dli_umol / 1_000_000.0
        daily_dlis[date] = dli_mol

    if not daily_dlis:
        return None

    daily_series = pd.Series(daily_dlis).sort_index()
    daily_series.index = pd.to_datetime(daily_series.index)
    daily_series.name = "DLI"
    return daily_series


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
            st.error(
                "Invalid username or password. If you do not have an account, please email greenhouseprofessors@gmail.com."
            )

    st.divider()
    st.caption("Email greenhouseprofessors@gmail.com to create an account.")
    st.caption("Beta program for the Floriculture Research Alliance")
    st.stop()

# ----- Logged-in view -----

# Top-row navigation & actions
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    if st.button("📂 Manual Upload", use_container_width=True):
        st.switch_page("pages/1_Upload.py")

with col2:
    if st.button("⚙️ Edit Settings", use_container_width=True):
        st.switch_page("pages/2_Settings.py")

with col3:
    # We fill this later once we’ve built the PDF.
    download_slot = st.empty()

# ----- Quick Upload panel (always visible) -----
st.markdown("### Quick Upload")
st.caption(
    "Drop a data file here to add it to your saved cleaned datasets. "
    "To edit column selections and preview data, use the Upload page. "
    "Edit units and desired conditions in Settings."
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
    file_id = (quick_file.name, quick_file.size)

    if st.session_state.last_quick_upload_file_id != file_id:
        original_name = quick_file.name
        ext = Path(original_name).suffix or ".csv"
        file_bytes_raw = quick_file.getvalue()

        try:
            # 1) Load raw table
            df_raw, file_type, encoding_used = load_table_from_bytes(file_bytes_raw, ext)

            # 2) Column mapping (use your saved Upload-page selections when possible)
            alias_table = build_alias_table()
            raw_cols = [str(c) for c in df_raw.columns]
            auto_mapping, _, _ = map_columns(raw_cols, alias_table)

            if not auto_mapping:
                raise ValueError("Could not automatically match any columns to Time/AirTemp/RH/PAR.")

            # Try to apply the user's LAST SAVED mapping template from the Upload page.
            prefs = db.get_last_upload_context(user["id"])
            template = prefs.get("canon_to_raw") or {}

            preferred_mapping = {}
            used = set()

            # First: apply template (canon -> raw) when the raw column exists in this file
            for canon, raw in template.items():
                if not raw:
                    continue
                if raw in raw_cols and raw not in used:
                    preferred_mapping[raw] = canon
                    used.add(raw)

            # Second: fill any missing canonicals using the automatic mapping
            if preferred_mapping:
                mapped_canons = set(preferred_mapping.values())
                for raw, canon in auto_mapping.items():
                    if canon in mapped_canons:
                        continue
                    if raw in used:
                        continue
                    preferred_mapping[raw] = canon
                    used.add(raw)

            mapping_to_use = preferred_mapping if preferred_mapping else auto_mapping

            # 3) Build cleaned dataframe
            df_clean = build_clean_dataframe(df_raw, mapping_to_use)

            ...

            cleaned_bytes = df_clean.to_csv(index=False).encode("utf-8")
            file_db_id = db.add_file_record(user["id"], stored_filename, cleaned_bytes)

            # Save mapping metadata so it can be reviewed/edited on the Upload page later
            try:
                canon_to_raw = {canon: None for canon in CANON_ORDER}
                for raw, canon in mapping_to_use.items():
                    canon_to_raw[canon] = raw

                db.upsert_file_column_map(
                    user["id"],
                    file_db_id,
                    raw_columns=raw_cols,
                    canon_to_raw=canon_to_raw,
                    raw_preview_rows=df_raw.head(10).to_dict(orient="records"),
                )
            except Exception:
                pass

            # 3) Build cleaned dataframe
            df_clean = build_clean_dataframe(df_raw, auto_mapping)

            required_for_dashboard = ["Time", "AirTemp", "RH"]
            missing_for_dashboard = [c for c in required_for_dashboard if c not in df_clean.columns]
            if missing_for_dashboard:
                raise ValueError("Missing required columns for dashboard: " + ", ".join(missing_for_dashboard))

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
                f"Quick upload succeeded and cleaned file `{stored_filename}` was saved."
            )

            st.session_state.last_quick_upload_file_id = file_id
            upload_succeeded = True

        except Exception as e:
            st.error(f"Quick upload could not automatically process this file: {e}")
            st.warning("Use the full Upload page to manually select columns and review the mapping for this dataset.")
            st.page_link("pages/1_Upload.py", label="⚠️ Unable to Upload File – Open Upload Page")

if upload_succeeded:
    st.rerun()

# ----- File selection (DIRECTLY BELOW Quick Upload) -----
st.markdown("### File Selection")

files = db.list_user_files(user["id"])
if not files:
    st.info("No cleaned files found yet. Upload a file above to get started.")
    st.stop()

options = {f"{rec['filename']} ({rec['uploaded_at']})": rec for rec in files}

# Persist selection across reruns; default to most recent
default_label = list(options.keys())[0]
if "selected_file_label" not in st.session_state:
    st.session_state.selected_file_label = default_label

selected_label = st.selectbox(
    "Select a cleaned data file",
    list(options.keys()),
    index=list(options.keys()).index(st.session_state.selected_file_label)
    if st.session_state.selected_file_label in options
    else 0,
    key="selected_file_label",
)

rec = options[selected_label]
st.session_state["selected_file_id"] = rec["id"]  # useful later for Chatbot, etc.

st.markdown("---")

# ----- Load selected file -----
file_obj = db.get_file_bytes(rec["id"])
if file_obj is None:
    st.error("Could not load the selected cleaned file from the database.")
    st.stop()

filename = file_obj["filename"]
file_bytes = file_obj["bytes"]

try:
    bio = io.BytesIO(file_bytes)
    df = pd.read_csv(bio)
except Exception as e:
    st.error(f"Could not read cleaned CSV from Neon: {e}")
    st.stop()

# Ensure Time is datetime & sorted
if "Time" in df.columns:
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time")

# ----- Load per-user settings -----
settings_row = db.get_or_create_settings(user["id"])
settings = dict(settings_row) if settings_row is not None else {}

# Units
temp_unit = settings.get("temp_unit", "F")  # display unit 'F' or 'C'
orig_temp_unit = settings.get("orig_temp_unit", None)  # 'C' or 'F'
orig_light_unit = settings.get("orig_light_unit", "PPFD")

# Targets
target_temp_low = float(settings.get("target_low", 65.0))
target_temp_high = float(settings.get("target_high", 85.0))

target_rh_low = float(settings.get("target_rh_low", 700.0))
target_rh_high = float(settings.get("target_rh_high", 95.0))

target_ppfd = float(settings.get("target_ppfd", 750.0))
target_dli = float(settings.get("target_dli", 10.0))

target_vpd_low = float(settings.get("target_vpd_low", 0.2))
target_vpd_high = float(settings.get("target_vpd_high", 1.0))

# ----- Core series -----
air_raw = df["AirTemp"].astype(float) if "AirTemp" in df.columns else None
leaf_raw = df["LeafTemp"].astype(float) if "LeafTemp" in df.columns else None
rh = df["RH"].astype(float) if "RH" in df.columns else None
par = df["PAR"].astype(float) if "PAR" in df.columns else None

# Determine original temperature units (prefer user setting; fallback to heuristic)
if orig_temp_unit in ("C", "F"):
    orig_is_f = (orig_temp_unit == "F")
else:
    orig_is_f = bool(air_raw is not None and air_raw.notna().any() and air_raw.mean(skipna=True) > 40.0)

# Convert to Celsius for physics / VPD
air_c = to_celsius(air_raw, orig_is_f) if air_raw is not None else None
leaf_c = to_celsius(leaf_raw, orig_is_f) if leaf_raw is not None else None

# VPD calculations (kPa)
vpd_air = None
vpd_leaf = None
leaf_air_diff_c = None

if air_c is not None and rh is not None:
    es_air = 0.61121 * np.exp((18.678 - air_c / 234.5) * (air_c / (257.14 + air_c)))
    ea_air = (rh / 100.0) * es_air
    vpd_air = (es_air - ea_air).clip(lower=0)

    if leaf_c is not None:
        es_leaf = 0.61121 * np.exp((18.678 - leaf_c / 234.5) * (leaf_c / (257.14 + leaf_c)))
        vpd_leaf = np.maximum(0, es_leaf - ea_air)
        leaf_air_diff_c = leaf_c - air_c

# Convert to display units
air_disp = to_display_temp(air_c, temp_unit) if air_c is not None else None
leaf_disp = to_display_temp(leaf_c, temp_unit) if leaf_c is not None else None
leaf_air_diff_disp = diff_to_display(leaf_air_diff_c, temp_unit) if leaf_air_diff_c is not None else None

df_display = df.copy()
if air_disp is not None:
    df_display["AirTemp"] = air_disp
if leaf_disp is not None:
    df_display["LeafTemp"] = leaf_disp

# Prefer leaf VPD if leaf temp exists; otherwise keep air VPD
if vpd_leaf is not None:
    df_display["VPDleaf"] = vpd_leaf
elif vpd_air is not None:
    df_display["VPDair"] = vpd_air

# Compute DLI only when PAR is truly PPFD
daily_dli_series = None
if orig_light_unit == "PPFD" and "Time" in df_display.columns and "PAR" in df_display.columns:
    daily_dli_series = compute_daily_dli(df_display[["Time", "PAR"]])

# =========================
# Dashboard Summary section
# =========================
st.subheader("Dashboard Summary")
st.caption(f"Showing selected file: `{rec['filename']}` (uploaded {rec['uploaded_at']}).")

# Summary sentence
start_time = None
end_time = None
interval_td = None
if "Time" in df_display.columns and df_display["Time"].notna().any():
    time_sorted = df_display["Time"].sort_values()
    start_time = time_sorted.iloc[0]
    end_time = time_sorted.iloc[-1]
    if len(time_sorted) >= 2:
        interval_td = time_sorted.iloc[1] - time_sorted.iloc[0]

    interval_str = format_timedelta(interval_td) if interval_td is not None else "unknown"
    st.caption(
        f"Data collection from **{start_time.strftime('%Y-%m-%d %H:%M:%S')}** "
        f"to **{end_time.strftime('%Y-%m-%d %H:%M:%S')}**, "
        f"with an approximate sampling interval of **{interval_str}**."
    )
else:
    st.caption("Time information is not available in this file.")

# Metric widgets
metric_cols = st.columns(4)

# 1) Average Air Temperature
within_pct = None
if air_disp is not None and air_disp.notna().any():
    air_mean = air_disp.mean(skipna=True)
    if air_mean > target_temp_high:
        delta_text = f"{air_mean - target_temp_high:.1f} above high target"
    elif air_mean < target_temp_low:
        delta_text = f"{target_temp_low - air_mean:.1f} below low target"
    else:
        delta_text = "Within target band"

    metric_cols[0].metric(
        label=f"Average Air Temperature ({'°F' if temp_unit == 'F' else '°C'})",
        value=f"{air_mean:.1f}",
        delta=delta_text,
    )

    temp_series = air_disp.dropna()
    if len(temp_series) > 0:
        within_mask = (temp_series >= target_temp_low) & (temp_series <= target_temp_high)
        within_pct = 100.0 * within_mask.mean()
else:
    metric_cols[0].write("Average Air Temperature: n/a")

# 2) Average Leaf Temperature
if leaf_disp is not None and leaf_disp.notna().any():
    leaf_mean = leaf_disp.mean(skipna=True)
    metric_cols[1].metric(
        label=f"Average Leaf Temperature ({'°F' if temp_unit == 'F' else '°C'})",
        value=f"{leaf_mean:.1f}",
    )
else:
    metric_cols[1].write("Average Leaf Temperature: n/a")

# 3) Average RH
if rh is not None and rh.notna().any():
    rh_mean = rh.mean(skipna=True)
    metric_cols[2].metric(
        label="Average Relative Humidity (%)",
        value=f"{rh_mean:.0f}",
    )
else:
    metric_cols[2].write("Average Relative Humidity: n/a")

# 4) % time within target temperature band
if within_pct is not None:
    metric_cols[3].metric(
        label="Time in target temperature band",
        value=f"{within_pct:.0f} %",
    )
else:
    metric_cols[3].write("Time in target band: n/a")

# Issue highlighting
if within_pct is not None:
    if within_pct < 50:
        st.error(f"Only about **{within_pct:.0f}%** of readings were within your target temperature band.")
    elif within_pct < 80:
        st.warning(f"About **{within_pct:.0f}%** of readings were within your target band.")
    else:
        st.success(f"About **{within_pct:.0f}%** of readings were within your target temperature band.")

# =========================
# Summary table moved HERE
# =========================
st.subheader("Summary Statistics")

numeric_cols = df_display.select_dtypes(include="number").columns.tolist()
summary = None

if numeric_cols:
    summary = df_display[numeric_cols].agg(["min", "mean", "max"]).transpose()
    summary.rename(columns={"min": "Min", "mean": "Average", "max": "Max"}, inplace=True)
    summary.index = [pretty_label(c, temp_unit) for c in summary.index]

    # Add DLI row if available
    if daily_dli_series is not None and not daily_dli_series.empty:
        dli_row = pd.DataFrame(
            {
                "Min": [daily_dli_series.min()],
                "Average": [daily_dli_series.mean()],
                "Max": [daily_dli_series.max()],
            },
            index=["Daily Light Integral (mol m⁻² d⁻¹)"],
        )
        summary = pd.concat([summary, dli_row], axis=0)

        # --- Display formatting: per-row decimals + PPFD Min/Average as "-" ---
        ppfd_label = "Light Intensity (PPFD - µmol m⁻² s⁻¹)"

        def row_format_spec(row_label: str) -> str:
            # Customize these however you want
            if "Relative Humidity" in row_label:
                return "{:.0f}"
            if "Vapor Pressure Deficit" in row_label:
                return "{:.2f}"
            if "Light Intensity" in row_label:
                return "{:.0f}"
            if "Daily Light Integral" in row_label:
                return "{:.1f}"
            if "Temperature" in row_label:
                return "{:.0f}"
            return "{:.1f}"  # default

        def fmt_cell(val, fmt: str) -> str:
            if pd.isna(val):
                return "-"
            try:
                return fmt.format(float(val))
            except Exception:
                return str(val)

        summary_display = summary.copy()

        # Force PPFD Min/Average to be "-"
        if ppfd_label in summary_display.index:
            summary_display.loc[ppfd_label, "Min"] = np.nan
            summary_display.loc[ppfd_label, "Average"] = np.nan

        # Convert to formatted strings (per-row format spec)
        for idx in summary_display.index:
            fmt = row_format_spec(str(idx))
            for col in summary_display.columns:
                # Keep the explicit "-" behavior for PPFD Min/Average
                if idx == ppfd_label and col in ["Min", "Average"]:
                    summary_display.at[idx, col] = "-"
                else:
                    summary_display.at[idx, col] = fmt_cell(summary_display.at[idx, col], fmt)

        st.dataframe(summary_display, use_container_width=True)

else:
    st.info("No numeric columns found to summarize.")

# =========================
# Key Trends (Dashboard graphs)
# =========================
st.markdown("### Key Trends")
st.subheader("Time series graphs")

use_time_axis = "Time" in df_display.columns and df_display["Time"].notna().any()
x_values = df_display["Time"] if use_time_axis else df_display.index

figs_for_pdf = []

# --- Cover page for PDF (data summary + targets) ---
fig_cover, ax_cover = plt.subplots(figsize=(8.5, 11))
ax_cover.axis("off")
title = "EnDash Dashboard Report"
ax_cover.text(0.5, 0.96, title, ha="center", va="top", fontsize=18, fontweight="bold")

ax_cover.text(0.05, 0.90, f"File: {filename}", fontsize=11)
ax_cover.text(0.05, 0.87, f"Uploaded: {rec['uploaded_at']}", fontsize=11)

if start_time is not None and end_time is not None:
    interval_str = format_timedelta(interval_td) if interval_td is not None else "unknown"
    ax_cover.text(0.05, 0.83, f"Time range: {start_time}  →  {end_time}", fontsize=11)
    ax_cover.text(0.05, 0.80, f"Approx. interval: {interval_str}", fontsize=11)

ax_cover.text(0.05, 0.74, "Targets", fontsize=13, fontweight="bold")
ax_cover.text(0.07, 0.70, f"Temperature band: {target_temp_low:.1f} to {target_temp_high:.1f} ({'°F' if temp_unit=='F' else '°C'})", fontsize=11)
ax_cover.text(0.07, 0.67, f"Relative humidity band: {target_rh_low:.0f}% to {target_rh_high:.0f}%", fontsize=11)
ax_cover.text(0.07, 0.64, f"PPFD target: {target_ppfd:.1f} µmol m⁻² s⁻¹", fontsize=11)
ax_cover.text(0.07, 0.61, f"DLI target: {target_dli:.1f} mol m⁻² d⁻¹", fontsize=11)
ax_cover.text(0.07, 0.58, f"VPD band: {target_vpd_low:.2f} to {target_vpd_high:.2f} kPa", fontsize=11)

if within_pct is not None:
    ax_cover.text(0.05, 0.50, "Summary", fontsize=13, fontweight="bold")
    ax_cover.text(0.07, 0.46, f"Time in target temperature band: {within_pct:.0f}%", fontsize=11)

figs_for_pdf.append(fig_cover)

# --- Summary-table page in PDF ---
if summary is not None:
    fig_summary, ax_summary = plt.subplots(figsize=(8.5, 4.5))
    ax_summary.axis("off")

    title_text = "Summary statistics"
    if start_time is not None and end_time is not None:
        t_text = f"Data from {start_time} to {end_time}"
        if interval_td is not None:
            t_text += f" | Interval: {format_timedelta(interval_td)}"
        title_text = f"{title_text}\n{t_text}"

    ax_summary.set_title(title_text, fontsize=10, pad=20)

    tbl = ax_summary.table(
        cellText=summary_display.values,   # <-- uses the same formatted strings (and "-" for PPFD)
        rowLabels=summary_display.index,
        colLabels=summary_display.columns,
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)

    figs_for_pdf.append(fig_summary)

# ---------- Special PPFD + DLI plot (dual Y-axes) ----------
numeric_cols_no_par = numeric_cols[:]
if "PAR" in numeric_cols:
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.set_ylabel("PPFD (µmol m⁻² s⁻¹)")
    ax1.set_title("Light Intensity and Daily Light Integral")
    ax2 = ax1.twinx()
    ax2.set_zorder(0)
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    ax1.plot(
        x_values,
        df_display["PAR"],
        color="tab:blue",
        zorder=2,
        label="PPFD (µmol m⁻² s⁻¹)",
    )

    ax1.axhline(
        target_ppfd,
        color="gold",
        linestyle="--",
        linewidth=1.0,
        zorder=2,
        label=f"Target PPFD ({target_ppfd:.1f})",
    )

    # DLI bars (only if computed)
    if daily_dli_series is not None and not daily_dli_series.empty and use_time_axis:
        dli_midpoints = daily_dli_series.index + pd.Timedelta(hours=12)
        ax2.bar(
            dli_midpoints,
            daily_dli_series.values,
            width=0.42,  # ~10 hours
            align="center",
            color="tab:orange",
            edgecolor="tab:orange",
            linewidth=0.5,
            zorder=0.5,
            alpha=1,
            label="DLI (mol m⁻² d⁻¹)",
        )
        ax2.axhline(
            target_dli,
            color="purple",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
            label=f"Target DLI ({target_dli:.1f})",
        )

    ax2.set_ylabel("DLI (mol m⁻² d⁻¹)")

    if use_time_axis:
        ax1.set_xlabel("Time")
        apply_time_axis_formatting(ax1, fig, x_values)
    else:
        ax1.set_xlabel("Index")
        ax1.xaxis.set_major_locator(plt.MaxNLocator(8))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.grid(True, linestyle=":", linewidth=0.5)

    st.pyplot(fig)
    figs_for_pdf.append(fig)

    numeric_cols_no_par = [c for c in numeric_cols if c != "PAR"]

# ---------- Generic plots for remaining numeric columns ----------
for col in numeric_cols_no_par:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x_values, df_display[col], label=pretty_label(col, temp_unit))

    if use_time_axis:
        ax.set_xlabel("Time")
        apply_time_axis_formatting(ax, fig, x_values)
    else:
        ax.set_xlabel("Index")
        ax.xaxis.set_major_locator(plt.MaxNLocator(8))

    y_label = pretty_label(col, temp_unit)
    ax.set_ylabel(y_label)
    ax.set_title(y_label)

    # Temperature target bands
    if col in ["AirTemp", "LeafTemp"]:
        ax.axhline(
            target_temp_high,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Target high temperature ({target_temp_high:.1f})",
        )
        ax.axhline(
            target_temp_low,
            color="blue",
            linestyle="--",
            linewidth=1.0,
            label=f"Target low temperature ({target_temp_low:.1f})",
        )


    # Relative humidity target band
    if col == "RH":
        ax.axhline(
            target_rh_high,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Target high Relative Humidity ({target_rh_high:.0f}%)",
        )
        ax.axhline(
            target_rh_low,
            color="blue",
            linestyle="--",
            linewidth=1.0,
            label=f"Target low Relative Humidity ({target_rh_low:.0f}%)",
        )
        

    # VPD target band
    if col in ["VPDair", "VPDleaf"]:
        ax.axhline(
            target_vpd_high,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Target high VPD ({target_vpd_high:.2f} kPa)",
        )
        ax.axhline(
            target_vpd_low,
            color="blue",
            linestyle="--",
            linewidth=1.0,
            label=f"Target low VPD ({target_vpd_low:.2f} kPa)",
        )
        

    ax.legend(loc="best")
    ax.grid(True, linestyle=":", linewidth=0.5)

    st.pyplot(fig)
    figs_for_pdf.append(fig)

# =========================
# Download Dashboard button (replaces old Full Dashboard)
# =========================
if figs_for_pdf:
    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        for fig in figs_for_pdf:
            pdf.savefig(fig, bbox_inches="tight")
    pdf_buffer.seek(0)

    # Put the download button in the TOP-RIGHT slot (replacing the old page button)
    download_slot.download_button(
        label="⬇️ Download Dashboard",
        data=pdf_buffer,
        file_name=f"endash_dashboard_{Path(filename).stem}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    # Close figures after rendering + PDF generation
    for fig in figs_for_pdf:
        plt.close(fig)
else:
    download_slot.button("⬇️ Download Dashboard", disabled=True, use_container_width=True)

st.markdown("---")
st.caption("Courtesy of the Fisher Lab - IFAS, University of Florida")
