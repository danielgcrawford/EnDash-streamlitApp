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

# Temperature display unit ('F' or 'C')
temp_unit = settings.get("temp_unit", "F")

# Temperature targets (for air & leaf plots)
target_temp_low = float(settings.get("target_low", 65.0))
target_temp_high = float(settings.get("target_high", 80.0))

# Relative humidity targets (%)
target_rh_low = float(settings.get("target_rh_low", 70.0))
target_rh_high = float(settings.get("target_rh_high", 95.0))

# Light & DLI targets
target_ppfd = float(settings.get("target_ppfd", 150.0))   # µmol m⁻² s⁻¹
target_dli = float(settings.get("target_dli", 8.0))       # mol m⁻² d⁻¹

# VPD targets (kPa)
target_vpd_low = float(settings.get("target_vpd_low", 0.2))
target_vpd_high = float(settings.get("target_vpd_high", 0.8))

# ----------------- Helper functions -----------------


def to_celsius(series: pd.Series, orig_is_fahrenheit: bool) -> pd.Series:
    """Convert a temperature series to °C if original was °F."""
    if series is None:
        return None
    return (series - 32.0) * 5.0 / 9.0 if orig_is_fahrenheit else series


def to_display_temp(series_c: pd.Series, unit: str) -> pd.Series:
    """Convert a °C series into the user-selected display unit (°C or °F)."""
    if series_c is None:
        return None
    if unit == "F":
        return series_c * 9.0 / 5.0 + 32.0
    return series_c


def diff_to_display(diff_c: pd.Series, unit: str) -> pd.Series:
    """
    Convert a temperature difference in °C into user-selected units.
    Note: a ΔT in °C and °F differ by factor 9/5 (no offset).
    """
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
    """
    Create a y-axis / table label with units, writing out full words
    instead of abbreviations.
    """
    temp_symbol = "°F" if temp_unit == "F" else "°C"
    if col == "AirTemp":
        return f"Air Temperature ({temp_symbol})"
    if col == "LeafTemp":
        return f"Leaf Temperature ({temp_symbol})"
    if col == "RH":
        return "Relative Humidity (%)"
    if col == "PAR":
        return "Photosynthetic Photon Flux Density (µmol m⁻² s⁻¹)"
    if col == "VPDleaf":
        return "Leaf Vapor Pressure Deficit (kPa)"
    if col == "VPDair":
        return "Air Vapor Pressure Deficit (kPa)"
    return col


def compute_daily_dli(df_light: pd.DataFrame) -> pd.Series | None:
    """
    Compute Daily Light Integral (DLI) from PPFD measurements.

    Inputs:
        df_light: DataFrame with columns:
            - 'Time': datetime64[ns]
            - 'PAR' : PPFD in µmol m⁻² s⁻¹

    Equation:
        DLI (mol m⁻² d⁻¹) = Σ (PPFD_i * Δt) / 1,000,000

        where:
            PPFD_i is the instantaneous PPFD (µmol m⁻² s⁻¹),
            Δt is the logging interval in seconds.

    Implementation details:
        - We assume a roughly constant logging interval and estimate Δt
          as the median difference between timestamps.
        - A "full day" is defined as a date where:
            * The time span of data for that date is >= 20 hours, AND
            * The number of samples is >= 80% of a full 24-hour day at
              the median logging interval.
        - Only such full days are included in the DLI series.
    """
    if "Time" not in df_light.columns or "PAR" not in df_light.columns:
        return None

    df_light = df_light.dropna(subset=["Time", "PAR"]).copy()
    if df_light.empty or df_light["Time"].nunique() < 2:
        return None

    df_light = df_light.sort_values("Time")

    # Estimate logging interval (seconds) from median time difference
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

        # Require near full-day coverage
        if span_sec < 20.0 * 3600.0:
            continue
        if n < 0.8 * expected_n:
            continue

        # DLI = sum(PPFD * Δt) / 1e6   (µmol → mol)
        dli_umol = group["PAR"].sum() * median_dt
        dli_mol = dli_umol / 1_000_000.0
        daily_dlis[date] = dli_mol

    if not daily_dlis:
        return None

    daily_series = pd.Series(daily_dlis).sort_index()
    daily_series.index = pd.to_datetime(daily_series.index)  # index = midnight of each date
    daily_series.name = "DLI"
    return daily_series


def apply_time_axis_formatting(ax, fig, x_values):
    """
    Apply reasonable tick spacing & formatting for a datetime x-axis
    based on the total span of the data.
    """
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
par = df["PAR"].astype(float) if "PAR" in df.columns else None

# Detect original temperature units from average air temperature
if air_raw is not None and air_raw.notna().any():
    # Simple threshold: > 40 => original is Fahrenheit
    orig_is_f = air_raw.mean(skipna=True) > 40.0
else:
    orig_is_f = False

# Convert to Celsius for all physics / VPD
air_c = to_celsius(air_raw, orig_is_f) if air_raw is not None else None
leaf_c = to_celsius(leaf_raw, orig_is_f) if leaf_raw is not None else None

# VPD calculations (in kPa, using °C for saturation vapor pressure)
vpd_air = None
vpd_leaf = None
leaf_air_diff_c = None

if air_c is not None and rh is not None:
    es_air = 0.61121 * np.exp((18.678 - air_c / 234.5) * (air_c / (257.14 + air_c)))
    ea_air = (rh / 100.0) * es_air
    vpd_air = es_air - ea_air  # kPa
    vpd_air = vpd_air.clip(lower=0)

    if leaf_c is not None:
        es_leaf = 0.61121 * np.exp(
            (18.678 - leaf_c / 234.5) * (leaf_c / (257.14 + leaf_c))
        )
        vpd_leaf = np.maximum(0, es_leaf - ea_air)
        leaf_air_diff_c = leaf_c - air_c

# Convert everything to user-selected display units
air_disp = to_display_temp(air_c, temp_unit) if air_c is not None else None
leaf_disp = to_display_temp(leaf_c, temp_unit) if leaf_c is not None else None
leaf_air_diff_disp = (
    diff_to_display(leaf_air_diff_c, temp_unit)
    if leaf_air_diff_c is not None
    else None
)

# Build display dataframe
df_display = df.copy()
if air_disp is not None:
    df_display["AirTemp"] = air_disp
if leaf_disp is not None:
    df_display["LeafTemp"] = leaf_disp
# If you later want a separate leaf-air diff column, you can add it here:
# if leaf_air_diff_disp is not None:
#     df_display["LeafAirDiff"] = leaf_air_diff_disp

if vpd_leaf is not None:
    df_display["VPDleaf"] = vpd_leaf
elif vpd_air is not None:
    df_display["VPDair"] = vpd_air

# Daily Light Integral (per full day), based on PPFD if available
daily_dli_series = None
if "Time" in df_display.columns and "PAR" in df_display.columns:
    daily_dli_series = compute_daily_dli(df_display[["Time", "PAR"]])

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
summary = None

if numeric_cols:
    # Base summary for all numeric columns in df_display
    summary = df_display[numeric_cols].agg(["min", "mean", "max"]).transpose()
    summary.rename(columns={"min": "Min", "mean": "Average", "max": "Max"}, inplace=True)

    # Use human-readable row labels
    new_index = [pretty_label(c) for c in summary.index]
    summary.index = new_index

    # Append DLI row (based on daily DLI series) if available
    if daily_dli_series is not None and not daily_dli_series.empty:
        dli_min = daily_dli_series.min()
        dli_mean = daily_dli_series.mean()
        dli_max = daily_dli_series.max()

        dli_row = pd.DataFrame(
            {
                "Min": [dli_min],
                "Average": [dli_mean],
                "Max": [dli_max],
            },
            index=["Daily Light Integral (mol m⁻² d⁻¹)"],
        )
        summary = pd.concat([summary, dli_row], axis=0)

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
if summary is not None:
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

# ---------- Special PPFD + DLI plot (dual Y-axes) ----------
if "PAR" in numeric_cols:
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.set_ylabel("PPFD (µmol m⁻² s⁻¹)")
    ax1.set_title("Light Intensity and Daily Light Integral")
    ax2 = ax1.twinx()
    ax2.set_zorder(0)
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    # Left axis: PPFD over time
    ax1.plot(
        x_values,
        df_display["PAR"],
        color="tab:blue",
        zorder=2,
        label="PPFD (µmol m⁻² s⁻¹)",
    )
   
    # Target PPFD line
    ax1.axhline(
        target_ppfd,
        color="gold",
        linestyle="--",
        linewidth=1.0,
        zorder=2,
        label=f"Target PPFD ({target_ppfd:.1f} µmol m⁻² s⁻¹)",
    )

    # Right axis: DLI by day (bars)

    if daily_dli_series is not None and not daily_dli_series.empty and use_time_axis:
        dli_dates = daily_dli_series.index
        dli_values = daily_dli_series.values
        dli_midpoints = dli_dates + pd.Timedelta(hours=12) #centers bars around mid-day
        
        # Width ~0.8 days so bars fill most of the day window
        ax2.bar(
            dli_midpoints,
            dli_values,
            width=0.42, #10 hours wide
            align="center",
            color="tab:orange",
            edgecolor="tab:orange",
            linewidth=0.5,
            zorder=0.5,
            alpha=1, #transparency
            label="DLI (mol m⁻² d⁻¹)",
        )

        # Target DLI line
        ax2.axhline(
            target_dli,
            color="purple",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
            label=f"Target DLI ({target_dli:.1f} mol m⁻² d⁻¹)",
        )

    ax2.set_ylabel("DLI (mol m⁻² d⁻¹)")

    # Shared x-axis formatting
    if use_time_axis:
        ax1.set_xlabel("Time")
        apply_time_axis_formatting(ax1, fig, x_values)
    else:
        ax1.set_xlabel("Index")
        ax1.xaxis.set_major_locator(plt.MaxNLocator(8))

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.grid(True, linestyle=":", linewidth=0.5)

    st.pyplot(fig)
    figs_for_pdf.append(fig)

# Remove PAR so it is not plotted again in the generic loop
numeric_cols_no_par = [c for c in numeric_cols if c != "PAR"]

# ---------- Generic plots for remaining numeric columns ----------
for col in numeric_cols_no_par:
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(x_values, df_display[col], label=pretty_label(col))

    if use_time_axis:
        ax.set_xlabel("Time")
        apply_time_axis_formatting(ax, fig, x_values)
    else:
        ax.set_xlabel("Index")
        ax.xaxis.set_major_locator(plt.MaxNLocator(8))

    y_label = pretty_label(col)
    ax.set_ylabel(y_label)
    ax.set_title(y_label)

    # Temperature target bands (air & leaf)
    if col in ["AirTemp", "LeafTemp"]:
        ax.axhline(
            target_temp_low,
            color="blue",
            linestyle="--",
            linewidth=1.0,
            label=f"Target low temperature ({target_temp_low:.1f})",
        )
        ax.axhline(
            target_temp_high,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Target high temperature ({target_temp_high:.1f})",
        )

    # Relative humidity target band
    if col == "RH":
        ax.axhline(
            target_rh_low,
            color="blue",
            linestyle="--",
            linewidth=1.0,
            label=f"Target low Relative Humidity ({target_rh_low:.0f}%)",
        )
        ax.axhline(
            target_rh_high,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Target high Relative Humidity ({target_rh_high:.0f}%)",
        )

    # VPD target band (applies to both air & leaf VPD)
    if col in ["VPDair", "VPDleaf"]:
        ax.axhline(
            target_vpd_low,
            color="blue",
            linestyle="--",
            linewidth=1.0,
            label=f"Target low VPD ({target_vpd_low:.2f} kPa)",
        )
        ax.axhline(
            target_vpd_high,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Target high VPD ({target_vpd_high:.2f} kPa)",
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

