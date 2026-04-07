# 2_Settings.py

import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from lib import auth, db


# ---------------------------------------------------------
# Page setup & auth
# ---------------------------------------------------------
st.set_page_config(
    page_title="Leaf Wetness",
    page_icon="🌿",
    layout="wide",
)
auth.require_login()
user = auth.current_user()
auth.render_sidebar()
db.init_db()

st.title("🌿 Leaf Wetness")
st.caption("Select the Leaf Wetness column for the current file, adjust Leaf Wetness settings, and view Leaf Wetness graphs.")


# ---------------------------------------------------------
# Helpers copied from existing app/upload logic
# ---------------------------------------------------------
def normalize(s: str) -> str:
    s = s.replace("\ufeff", "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


ALIASES = {
    "Time": ["time", "timestamp", "date_time", "datetime", "recorded at", "date.time", "logtime", "measurement_time"],
    "AirTemp": ["airtemp", "air_temp", "tair", "t_air", "ambient_temp", "air temperature", "air temperature (c)", "ta_c", "rhttemperature", "RHT - Temperature", "rhttemp", "RHT-Temperature"],
    "LeafTemp": ["leaftemp", "leaf_temp", "tleaf", "leaf temperature", "canopy_temp", "tc_leaf", "leaf_t (c)", "leaf_tc"],
    "RH": ["rel_hum", "relative_humidity", "humidity", "rh (%)", "rhhumidity", "rht_humidity", "rh_percent"],
    "PAR": ["par", "ppfd", "photosynthetically active radiation", "par_umol", "par (umol m-2 s-1)", "par_umolm2s", "quantum", "quantum_sensor", "quantumsensor", "quantumpar"],
    "Irrigation1": ["irrigation", "irrigation1", "irrigation_1", "irrig_1", "zone1", "valve1", "mist1"],
    "Irrigation2": ["irrigation2", "irrigation_2", "irrig_2", "zone2", "valve2", "mist2"],
    "Irrigation3": ["irrigation3", "irrigation_3", "irrig_3", "zone3", "valve3", "mist3"],
    "Irrigation4": ["irrigation4", "irrigation_4", "irrig_4", "zone4", "valve4", "mist4"],
    "Irrigation5": ["irrigation5", "irrigation_5", "irrig_5", "zone5", "valve5", "mist5"],
    "LeafWetness": ["leaf wetness", "leafwetness", "leaf_wetness", "lw", "leaf wetness %", "leaf wetness (%)", "leaf wetness (v)"],
    "Date": ["date", "day", "log_date", "recorded_date"],
    "TimeOfDay": ["time_of_day", "clock", "clock_time", "timeofday", "time only", "time_only"],
}

MAX_IRRIGATION_ZONES = 5
CANON_OUTPUT_BASE = ["Time", "AirTemp", "LeafTemp", "RH", "PAR", "LeafWetness"]
IRR_CANONS = [f"Irrigation{i}" for i in range(1, MAX_IRRIGATION_ZONES + 1)]
CANON_OUTPUT_ORDER = CANON_OUTPUT_BASE + IRR_CANONS


def build_alias_table():
    table = {}
    for canon, aliases in ALIASES.items():
        table[canon] = {normalize(canon), *[normalize(a) for a in aliases]}
    return table


def combine_date_time(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    d = pd.to_datetime(date_s, errors="coerce")
    if pd.api.types.is_numeric_dtype(time_s):
        tnum = pd.to_numeric(time_s, errors="coerce")
        return d.dt.normalize() + pd.to_timedelta(tnum, unit="D")
    t = pd.to_datetime(time_s, errors="coerce")
    td = t - t.dt.normalize()
    return d.dt.normalize() + td


def build_clean_dataframe(df_raw: pd.DataFrame, raw_to_canon: Dict[str, str]) -> pd.DataFrame:
    canon_to_raw: Dict[str, str] = {}
    for raw, canon in raw_to_canon.items():
        canon_to_raw.setdefault(canon, raw)

    data = {}

    raw_time = canon_to_raw.get("Time")
    raw_date = canon_to_raw.get("Date")
    raw_timeofday = canon_to_raw.get("TimeOfDay")

    time_series = None
    if raw_date and raw_date in df_raw.columns and (
        (raw_timeofday and raw_timeofday in df_raw.columns) or (raw_time and raw_time in df_raw.columns)
    ):
        tcol = raw_timeofday if (raw_timeofday and raw_timeofday in df_raw.columns) else raw_time
        time_series = combine_date_time(df_raw[raw_date], df_raw[tcol])
    elif raw_time and raw_time in df_raw.columns:
        time_series = pd.to_datetime(df_raw[raw_time], errors="coerce")

    if time_series is not None:
        data["Time"] = time_series

    for canon in CANON_OUTPUT_ORDER:
        if canon == "Time":
            continue
        raw = canon_to_raw.get(canon)
        if raw is None or raw not in df_raw.columns:
            continue
        data[canon] = pd.to_numeric(df_raw[raw], errors="coerce")

    if not data:
        return pd.DataFrame()

    df_clean = pd.DataFrame(data)
    if "Time" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["Time"]).sort_values("Time")
    return df_clean.dropna(axis=0, how="all").drop_duplicates()


def load_table_from_bytes(file_bytes: bytes, ext: str):
    ext = ext.lower()
    if ext in [".xlsx", ".xls", ".xlsm"]:
        bio = io.BytesIO(file_bytes)
        return pd.read_excel(bio), "excel", None, 0

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            bio = io.BytesIO(file_bytes)
            df = pd.read_csv(bio, encoding=enc)
            return df, "csv", enc, 0
        except Exception as e:
            last_err = e
    raise last_err


def find_full_days(time_s: pd.Series) -> list[pd.Timestamp]:
    t = pd.to_datetime(time_s, errors="coerce").dropna().sort_values()
    if t.nunique() < 2:
        return []

    dt_sec = t.diff().dt.total_seconds().dropna()
    if dt_sec.empty:
        return []

    median_dt = float(dt_sec.median())
    if median_dt <= 0:
        return []

    seconds_per_day = 24.0 * 3600.0
    df_t = pd.DataFrame({"Time": t})
    df_t["Date"] = df_t["Time"].dt.date

    full = []
    for d, g in df_t.groupby("Date"):
        n = len(g)
        span_sec = (g["Time"].iloc[-1] - g["Time"].iloc[0]).total_seconds()
        expected_n = seconds_per_day / median_dt
        if span_sec >= 20.0 * 3600.0 and n >= 0.8 * expected_n:
            full.append(pd.to_datetime(d))

    return sorted(full)


def detect_leaf_wetness_event_times(time_s: pd.Series, lw_s: pd.Series, sensitivity_pct: float, min_gap_min: float):
    t = pd.to_datetime(time_s, errors="coerce")
    lw = pd.to_numeric(lw_s, errors="coerce")

    prev = lw.shift(1)
    denom = prev.where(prev.abs() > 1e-9)
    rise_pct = ((lw - prev) / denom) * 100.0

    cand = t[(rise_pct > float(sensitivity_pct)) & rise_pct.notna()].dropna().sort_values()
    if cand.empty:
        return pd.DatetimeIndex([])

    if float(min_gap_min) <= 0:
        return pd.DatetimeIndex(cand)

    gap = pd.Timedelta(minutes=float(min_gap_min))
    kept = [cand.iloc[0]]
    last = kept[0]
    for cur in cand.iloc[1:]:
        if (cur - last) >= gap:
            kept.append(cur)
            last = cur
    return pd.DatetimeIndex(kept)


def apply_time_axis_formatting(ax, fig, x_values):
    time_min = x_values.min()
    time_max = x_values.max()
    total_seconds = (time_max - time_min).total_seconds()

    if total_seconds <= 6 * 3600:
        locator = mdates.MinuteLocator(interval=10)
        formatter = mdates.DateFormatter("%H:%M")
    elif total_seconds <= 24 * 3600:
        locator = mdates.HourLocator(interval=2)
        formatter = mdates.DateFormatter("%H:%M")
    elif total_seconds <= 7 * 24 * 3600:
        locator = mdates.DayLocator(interval=1)
        formatter = mdates.DateFormatter("%m-%d")
    else:
        locator = mdates.AutoDateLocator()
        formatter = mdates.AutoDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=30, ha="right")


def legend_below(ax, fig, ncol=2, y=-0.50):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=True,
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.28)


# ---------------------------------------------------------
# File selection
# ---------------------------------------------------------
files = db.list_user_files(user["id"])
if not files:
    st.info("No cleaned files found yet.")
    st.stop()

id_to_rec = {int(rec["id"]): rec for rec in files}
file_ids = list(id_to_rec.keys())

def _file_label(fid: int) -> str:
    r = id_to_rec[int(fid)]
    return f"{r['filename']} ({r['uploaded_at']})"

selected_file_id = st.selectbox(
    "Current File Selection",
    file_ids,
    format_func=_file_label,
    key="leaf_wetness_selected_file_id",
)

rec = id_to_rec[int(selected_file_id)]

# ---------------------------------------------------------
# Load cleaned file + saved mapping
# ---------------------------------------------------------
file_obj = db.get_file_bytes(rec["id"])
if file_obj is None:
    st.error("Could not load the selected cleaned file.")
    st.stop()

df = pd.read_csv(io.BytesIO(file_obj["bytes"]))
if "Time" in df.columns:
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time")

settings_row = db.get_or_create_settings(user["id"])
settings = dict(settings_row) if settings_row is not None else {}

leaf_wetness_unit = settings.get("leaf_wetness_unit", "Percent")
irrigation_sensitivity_pct = float(settings.get("irrigation_sensitivity_pct", 3.0))
leaf_wetness_min_interval_min = float(settings.get("leaf_wetness_min_interval_min", 7.0))
irrigation_trigger = float(settings.get("irrigation_trigger", 1.0))
irrigation_min_interval_min = float(settings.get("irrigation_min_interval_min", 7.0))
water_applied_per_event_ml_m2 = float(settings.get("water_applied_per_event_ml_m2", 10.0))

fcm = db.get_file_column_map(int(rec["id"])) or {}
canon_to_raw_saved = fcm.get("canon_to_raw") or {}

raw_obj = db.get_raw_file_bytes(int(rec["id"]))
if raw_obj is None:
    st.error("This file does not have raw upload bytes saved, so Leaf Wetness mapping cannot be edited for it.")
    st.stop()

raw_filename = raw_obj["raw_filename"]
ext = Path(raw_filename).suffix or ".csv"
df_raw, _, _, _ = load_table_from_bytes(raw_obj["bytes"], ext)
raw_cols = [str(c) for c in df_raw.columns]

# ---------------------------------------------------------
# Leaf Wetness raw-column selector
# ---------------------------------------------------------
st.markdown("### Leaf Wetness Data Column")

lw_options = ["(None)"] + raw_cols
current_lw_raw = canon_to_raw_saved.get("LeafWetness")
lw_index = lw_options.index(current_lw_raw) if current_lw_raw in lw_options else 0

selected_lw_raw = st.selectbox(
    "Leaf Wetness column",
    options=lw_options,
    index=lw_index,
    help="Select the raw uploaded-file column that should be treated as Leaf Wetness for this file.",
)

if st.button("💾 Save Leaf Wetness Column", key="save_leaf_wetness_column"):
    new_canon_to_raw = dict(canon_to_raw_saved)
    new_canon_to_raw["LeafWetness"] = None if selected_lw_raw == "(None)" else selected_lw_raw

    db.upsert_file_column_map(
        user["id"],
        int(rec["id"]),
        raw_columns=raw_cols,
        canon_to_raw=new_canon_to_raw,
        raw_preview_rows=None,
    )

    raw_to_canon = {raw: canon for canon, raw in new_canon_to_raw.items() if raw}
    df_clean2 = build_clean_dataframe(df_raw, raw_to_canon)

    required_for_dashboard = ["Time", "AirTemp", "RH"]
    missing = [c for c in required_for_dashboard if c not in df_clean2.columns]
    if missing:
        st.error(
            "Cannot apply this mapping because the cleaned file would be missing required columns: "
            + ", ".join(missing)
        )
        st.stop()

    db.update_file_content(int(rec["id"]), df_clean2.to_csv(index=False).encode("utf-8"))
    st.success("Leaf Wetness column saved.")
    st.rerun()

# reload cleaned file after possible rerun state
file_obj = db.get_file_bytes(rec["id"])
df = pd.read_csv(io.BytesIO(file_obj["bytes"]))
if "Time" in df.columns:
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time")

# ---------------------------------------------------------
# Leaf Wetness settings
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### Leaf Wetness Settings")

with st.form("leaf_wetness_settings_form", clear_on_submit=False):
    col_u, col_s, col_g = st.columns(3)

    with col_u:
        lw_unit_input = st.selectbox(
            "Leaf Wetness unit",
            options=["Percent", "Volts", "milliVolts"],
            index=["Percent", "Volts", "milliVolts"].index(
                leaf_wetness_unit if leaf_wetness_unit in ["Percent", "Volts", "milliVolts"] else "Percent"
            ),
        )

    with col_s:
        irrigation_sensitivity_pct_input = st.number_input(
            "Irrigation Sensitivity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(irrigation_sensitivity_pct),
            step=0.1,
        )

    with col_g:
        leaf_wetness_min_interval_input = st.number_input(
            "Minimum Time Between Irrigation Events (minutes)",
            value=float(leaf_wetness_min_interval_min),
            min_value=0.0,
            step=1.0,
            format="%.0f",
        )

    save_lw_settings = st.form_submit_button("💾 Save Leaf Wetness Settings")

if save_lw_settings:
    db.update_leaf_wetness_event_settings(
        user["id"],
        irrigation_sensitivity_pct=float(irrigation_sensitivity_pct_input),
        leaf_wetness_min_interval_min=float(leaf_wetness_min_interval_input),
    )

    db.update_settings(
        user["id"],
        orig_temp_unit=settings.get("orig_temp_unit", "C"),
        orig_light_unit=settings.get("orig_light_unit", "PPFD"),
        temp_unit=settings.get("temp_unit", "F"),
        target_low=float(settings.get("target_low", 65.0)),
        target_high=float(settings.get("target_high", 80.0)),
        target_rh_low=float(settings.get("target_rh_low", 70.0)),
        target_rh_high=float(settings.get("target_rh_high", 95.0)),
        target_ppfd=float(settings.get("target_ppfd", 150.0)),
        target_dli_low=float(settings.get("target_dli_low", 8.0)),
        target_dli_high=float(settings.get("target_dli_high", 12.0)),
        target_vpd_low=float(settings.get("target_vpd_low", 0.2)),
        target_vpd_high=float(settings.get("target_vpd_high", 0.8)),
        irrigation_trigger=float(irrigation_trigger),
        irrigation_min_interval_min=float(irrigation_min_interval_min),
        leaf_wetness_unit=lw_unit_input,
        irrigation_sensitivity_pct=float(irrigation_sensitivity_pct_input),
        leaf_wetness_min_interval_min=float(leaf_wetness_min_interval_input),
        water_applied_per_event_ml_m2=float(water_applied_per_event_ml_m2),
    )

    st.success("Leaf Wetness settings saved.")
    st.rerun()

# refresh settings after save
settings_row = db.get_or_create_settings(user["id"])
settings = dict(settings_row) if settings_row is not None else {}
leaf_wetness_unit = settings.get("leaf_wetness_unit", "Percent")
irrigation_sensitivity_pct = float(settings.get("irrigation_sensitivity_pct", 3.0))
leaf_wetness_min_interval_min = float(settings.get("leaf_wetness_min_interval_min", 7.0))
water_applied_per_event_ml_m2 = float(settings.get("water_applied_per_event_ml_m2", 10.0))

# ---------------------------------------------------------
# Derived Leaf Wetness outputs
# ---------------------------------------------------------
if "LeafWetness" not in df.columns or "Time" not in df.columns:
    st.warning("This file does not currently have a mapped Leaf Wetness column.")
    st.stop()

LEAF_WETNESS_EVENT_COL = "IrrigationEvents_LeafWetness"
df[LEAF_WETNESS_EVENT_COL] = 0

lw_event_times = detect_leaf_wetness_event_times(
    df["Time"],
    df["LeafWetness"],
    sensitivity_pct=irrigation_sensitivity_pct,
    min_gap_min=leaf_wetness_min_interval_min,
)

if len(lw_event_times) > 0:
    df.loc[df["Time"].isin(lw_event_times), LEAF_WETNESS_EVENT_COL] = 1

leafwetness_daily_counts = None
full_days = find_full_days(df["Time"])
if full_days:
    df_full = df[df["Time"].dt.normalize().isin(full_days)].copy()
    daily = df_full[df_full[LEAF_WETNESS_EVENT_COL] == 1].groupby(df_full["Time"].dt.normalize()).size()
    leafwetness_daily_counts = daily.reindex(full_days, fill_value=0)

LEAF_WETNESS_YLABEL = {
    "Percent": "Leaf Wetness (%)",
    "Volts": "Leaf Wetness (V)",
    "milliVolts": "Leaf Wetness (mV)",
}.get(leaf_wetness_unit, "Leaf Wetness")

# ---------------------------------------------------------
# Leaf Wetness summary table
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### Leaf Wetness Summary")

rows = []

lw_series = pd.to_numeric(df["LeafWetness"], errors="coerce").dropna()
if not lw_series.empty:
    rows.append({
        "Metric": "Leaf Wetness",
        "Min": lw_series.min(),
        "Average": lw_series.mean(),
        "Max": lw_series.max(),
    })

if leafwetness_daily_counts is not None and not leafwetness_daily_counts.empty:
    rows.append({
        "Metric": "Irrigation Events per Day (#) - Leaf Wetness",
        "Min": leafwetness_daily_counts.min(),
        "Average": leafwetness_daily_counts.mean(),
        "Max": leafwetness_daily_counts.max(),
    })

    if float(water_applied_per_event_ml_m2) > 0:
        wapd_lw = leafwetness_daily_counts.astype(float) * float(water_applied_per_event_ml_m2)
        rows.append({
            "Metric": "Water Applied per Day (mL/m²·day) - Leaf Wetness",
            "Min": wapd_lw.min(),
            "Average": wapd_lw.mean(),
            "Max": wapd_lw.max(),
        })

if rows:
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, width="stretch", hide_index=True)
else:
    st.info("No Leaf Wetness summary values are available yet.")

# ---------------------------------------------------------
# Leaf Wetness graphs
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### Leaf Wetness Graphs")

full_days_lw = find_full_days(df["Time"])
lw_day_options = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in full_days_lw] if full_days_lw else []

lw_day_key = "leaf_wetness_page_day_to_graph"
if lw_day_options:
    if (lw_day_key not in st.session_state) or (st.session_state[lw_day_key] not in lw_day_options):
        st.session_state[lw_day_key] = lw_day_options[-1]

    st.selectbox(
        "Day to Graph",
        options=lw_day_options,
        key=lw_day_key,
        help="Select a full day to display the 24hr Leaf Wetness graph.",
    )

    day_to_plot_lw = pd.to_datetime(st.session_state[lw_day_key])
    df_lw_day = df[df["Time"].dt.normalize() == day_to_plot_lw.normalize()].copy()

    if not df_lw_day.empty:
        fig_lw, ax_lw = plt.subplots(figsize=(8, 3))
        ax_lw.plot(df_lw_day["Time"], df_lw_day["LeafWetness"], label="Leaf Wetness")

        ev_mask = df_lw_day[LEAF_WETNESS_EVENT_COL] == 1
        if ev_mask.any():
            ax_lw.scatter(
                df_lw_day.loc[ev_mask, "Time"],
                df_lw_day.loc[ev_mask, "LeafWetness"],
                label="Irrigation Event (Leaf Wetness)",
                color="tab:orange",
                zorder=5,
            )

        ax_lw.set_title(f"Leaf Wetness (24hr) — {day_to_plot_lw.date()}")
        ax_lw.set_xlabel("Time of day")
        ax_lw.set_ylabel(LEAF_WETNESS_YLABEL)
        apply_time_axis_formatting(ax_lw, fig_lw, df_lw_day["Time"])
        legend_below(ax_lw, fig_lw, ncol=2, y=-0.5)
        st.pyplot(fig_lw)

if leafwetness_daily_counts is not None and len(leafwetness_daily_counts) > 0:
    fig_lw_bar, ax_lw_bar = plt.subplots(figsize=(8, 3))
    x_bar = pd.to_datetime(leafwetness_daily_counts.index) + pd.Timedelta(hours=12)

    ax_lw_bar.bar(x_bar, leafwetness_daily_counts.values, width=0.8, align="center")
    ax_lw_bar.set_title("Irrigation Events per Day (Leaf Wetness) — Full Days Only")
    ax_lw_bar.set_xlabel("Date")
    ax_lw_bar.set_ylabel("Events per day")
    apply_time_axis_formatting(ax_lw_bar, fig_lw_bar, pd.to_datetime(leafwetness_daily_counts.index))
    st.pyplot(fig_lw_bar)