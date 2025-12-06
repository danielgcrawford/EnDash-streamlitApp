# pages/3_Dashboard.py

import io
from io import BytesIO
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

from lib import auth, db

# ----------------- Page setup & auth -----------------

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
auth.require_login()
user = auth.current_user()
auth.render_sidebar()

st.title("📊 Dashboard")

# ----------------- File selection (cleaned files from Neon) -----------------

files = db.list_user_files(user["id"])
if not files:
    st.info("No cleaned files found. Please upload a file on the **Upload** page first.")
    st.stop()

options = {f"{rec['filename']} ({rec['uploaded_at']})": rec for rec in files}
label = st.selectbox("Select a cleaned data file", list(options.keys()))
rec = options[label]

file_id = rec["id"]
file_obj = db.get_file_bytes(file_id)
if file_obj is None:
    st.error("Could not load file from database.")
    st.stop()

filename = file_obj["filename"]
file_bytes = file_obj["bytes"]

try:
    bio = io.BytesIO(file_bytes)
    df = pd.read_csv(bio)
except Exception as e:
    st.error(f"Could not read cleaned CSV from Neon: {e}")
    st.stop()

# --- Quick preview to verify cleaned file structure ---
st.subheader("Cleaned file preview")
st.caption("First 10 rows from the cleaned CSV stored in Neon.")
st.dataframe(df.head(10), use_container_width=True)

cols_str = ", ".join(map(str, df.columns))
st.caption(f"Detected columns: {cols_str}")
st.divider()

# Ensure Time column is datetime and sorted, if present
if "Time" in df.columns:
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time")

# ----------------- Load per-user settings -----------------

settings_row = db.get_or_create_settings(user["id"])
settings = dict(settings_row) if settings_row is not None else {}

temp_unit = settings.get("temp_unit", "F")        # 'F' or 'C' (user preference)
target_low = float(settings.get("target_low", 65))
target_high = float(settings.get("target_high", 80))

# ----------------- Helper functions -----------------


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


def pretty_label(col: str) -> str:
    """Create a y-axis / table label with units."""
    temp_symbol = "°F" if temp_unit == "F" else "°C"
    if col == "AirTemp":
        return f"Air temperature ({temp_symbol})"
    if col == "LeafTemp":
        return f"Leaf temperature ({temp_symbol})"
    #if col == "LeafAirDiff":
        #return f"Leaf–air temperature difference ({temp_symbol})"
    if col == "RH":
        return "Relative humidity (%)"
    if col == "PAR":
        return "PAR (µmol m⁻² s⁻¹)"
    if col == "VPDleaf":
        return "Leaf VPD (kPa)"
    else:
        if col == "VPDair":
            return "Air VPD (kPa)"
    return col


# ----------------- Core calculations -----------------

required_cols = ["AirTemp", "LeafTemp", "RH"]
missing_core = [c for c in required_cols if c not in df.columns]
if missing_core:
    st.warning(
        "The cleaned file is missing required columns for all calculations: "
        + ", ".join(missing_core)
        + ". Some metrics may be skipped."
    )

# Get raw series if present
air_raw = df["AirTemp"].astype(float) if "AirTemp" in df.columns else None
leaf_raw = df["LeafTemp"].astype(float) if "LeafTemp" in df.columns else None
rh = df["RH"].astype(float) if "RH" in df.columns else None

# Detect original temperature units from average air temperature
if air_raw is not None and air_raw.notna().any():
    # Simple threshold: > 40 => original is Fahrenheit
    orig_is_f = air_raw.mean(skipna=True) > 40.0
else:
    orig_is_f = False

# Convert to Celsius for all physics / VPD
air_c = to_celsius(air_raw, orig_is_f) if air_raw is not None else None
leaf_c = to_celsius(leaf_raw, orig_is_f) if leaf_raw is not None else None

# VPD calculations
vpd_air = None
vpd_leaf = None
leaf_air_diff_c = None

if air_c is not None and rh is not None:
    es_air = 0.61121 * np.exp((18.678 - air_c / 234.5) * (air_c / (257.14 + air_c)))
    ea_air = (rh / 100.0) * es_air
    vpd_air = es_air - ea_air  # kPa
    vpd_air = vpd_air.clip(lower=0)

    if leaf_c is not None:
        es_leaf = 0.61121 * np.exp((18.678 - leaf_c / 234.5) * (leaf_c / (257.14 + leaf_c)))
        vpd_leaf = np.maximum(0, es_leaf - ea_air)
        leaf_air_diff_c = leaf_c - air_c

# Convert everything to user-selected display units
air_disp = to_display_temp(air_c, temp_unit) if air_c is not None else None
leaf_disp = to_display_temp(leaf_c, temp_unit) if leaf_c is not None else None
leaf_air_diff_disp = diff_to_display(leaf_air_diff_c, temp_unit) if leaf_air_diff_c is not None else None

# Build display dataframe
df_display = df.copy()
if air_disp is not None:
    df_display["AirTemp"] = air_disp
if leaf_disp is not None:
    df_display["LeafTemp"] = leaf_disp
#if leaf_air_diff_disp is not None:
#    df_display["LeafAirDiff"] = leaf_air_diff_disp
if vpd_leaf is not None:
    df_display["VPDleaf"] = vpd_leaf
else:
    if vpd_air is not None:
        df_display["VPDair"] = vpd_air
# ----------------- Summary sentence -----------------

if "Time" in df_display.columns and df_display["Time"].notna().any():
    time_sorted = df_display["Time"].sort_values()
    start_time = time_sorted.iloc[0]
    end_time = time_sorted.iloc[-1]
    interval_str = "unknown"
    interval_td = None
    if len(time_sorted) >= 2:
        interval_td = time_sorted.iloc[1] - time_sorted.iloc[0]
        interval_str = format_timedelta(interval_td)

    st.markdown(
        f"Data collection from **{start_time.strftime('%Y-%m-%d %H:%M:%S')}** "
        f"to **{end_time.strftime('%Y-%m-%d %H:%M:%S')}**, "
        f"with an approximate sampling interval of **{interval_str}**."
    )
else:
    st.markdown("Time information is not available in this file.")

st.divider()

# ----------------- Summary table -----------------

st.subheader("Summary Statistics")

numeric_cols = df_display.select_dtypes(include="number").columns.tolist()
if numeric_cols:
    summary = df_display[numeric_cols].agg(["min", "mean", "max"]).transpose()
    summary.rename(columns={"min": "Min", "mean": "Average", "max": "Max"}, inplace=True)

    new_index = [pretty_label(c) for c in summary.index]
    summary.index = new_index

    st.dataframe(
        summary.style.format("{:.1f}"),
        use_container_width=True,
    )
else:
    st.info("No numeric columns found to summarize.")

st.divider()

# ----------------- Time series plots -----------------

st.subheader("Time series graphs")

use_time_axis = "Time" in df_display.columns and df_display["Time"].notna().any()
x_values = df_display["Time"] if use_time_axis else df_display.index

figs_for_pdf = []

# Summary-table page in PDF
if numeric_cols:
    fig_summary, ax_summary = plt.subplots(figsize=(8.5, 4.5))
    ax_summary.axis("off")

    title_text = "Summary statistics"
    if "start_time" in locals() and "end_time" in locals():
        t_text = (
            f"Data from {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"to {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if interval_td is not None:
            t_text += f" | Interval: {format_timedelta(interval_td)}"
        title_text = f"{title_text}\n{t_text}"

    ax_summary.set_title(title_text, fontsize=10, pad=20)

    tbl = ax_summary.table(
        cellText=summary.round(1).values,
        rowLabels=summary.index,
        colLabels=summary.columns,
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)

    figs_for_pdf.append(fig_summary)

# One plot per numeric column
for col in numeric_cols:
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(x_values, df_display[col], label=pretty_label(col))

    if use_time_axis:
        ax.set_xlabel("Time")

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
    else:
        ax.set_xlabel("Index")
        ax.xaxis.set_major_locator(plt.MaxNLocator(8))

    y_label = pretty_label(col)
    ax.set_ylabel(y_label)
    ax.set_title(y_label)

    if col in ["AirTemp", "LeafTemp"]:
        ax.axhline(
            target_low,
            color="blue",
            linestyle="--",
            linewidth=1.0,
            label=f"Target low ({target_low:.1f})",
        )
        ax.axhline(
            target_high,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Target high ({target_high:.1f})",
        )

    ax.legend(loc="best")
    ax.grid(True, linestyle=":", linewidth=0.5)

    st.pyplot(fig)
    figs_for_pdf.append(fig)

# ----------------- PDF download -----------------

if figs_for_pdf:
    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        for fig in figs_for_pdf:
            pdf.savefig(fig, bbox_inches="tight")
    pdf_buffer.seek(0)

    st.download_button(
        label="⬇️ Download dashboard report (PDF)",
        data=pdf_buffer,
        file_name=f"dashboard_report_{Path(filename).stem}.pdf",
        mime="application/pdf",
    )

    for fig in figs_for_pdf:
        plt.close(fig)
