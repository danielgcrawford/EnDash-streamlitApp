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

import csv
from io import StringIO

st.set_page_config(
    page_title="EnDash - Climate Data Analysis",
    page_icon="🌿",
    layout="wide",
)

#st.logo("assets/EnDash_Logo_V1.png")

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
    "Time": ["time", "timestamp", "date_time", "datetime", "recorded at", "date.time", "logtime", "measurement_time",],
    "AirTemp": ["airtemp", "air_temp", "tair", "t_air", "ambient_temp", "air temperature", "air temperature (c)", "ta_c", "rhttemperature", "RHT - Temperature", "rhttemp", "RHT-Temperature",],
    "LeafTemp": ["leaftemp", "leaf_temp", "tleaf", "leaf temperature", "canopy_temp", "tc_leaf", "leaf_t (c)", "leaf_tc",],
    "RH": ["rel_hum", "relative_humidity", "humidity", "rh (%)", "rhhumidity", "rht_humidity", "rh_percent",],
    "PAR": ["par", "ppfd", "photosynthetically active radiation", "par_umol", "par (umol m-2 s-1)", "par_umolm2s", "quantum", "quantum_sensor", "quantumsensor", "quantumpar",],
    "Irrigation1": ["irrigation", "irrigation1", "irrigation_1", "irrig_1", "zone1", "valve1", "mist1"],
    "Irrigation2": ["irrigation2", "irrigation_2", "irrig_2", "zone2", "valve2", "mist2"],
    "Irrigation3": ["irrigation3", "irrigation_3", "irrig_3", "zone3", "valve3", "mist3"],
    "Irrigation4": ["irrigation4", "irrigation_4", "irrig_4", "zone4", "valve4", "mist4"],
    "Irrigation5": ["irrigation5", "irrigation_5", "irrig_5", "zone5", "valve5", "mist5"],
    "LeafWetness": ["leaf wetness", "leafwetness", "leaf_wetness", "lw","leaf wetness %", "leaf wetness (%)", "leaf wetness (v)"],
    "Date": ["date", "day", "log_date", "recorded_date"],
    "TimeOfDay": ["time_of_day", "clock", "clock_time", "timeofday", "time only", "time_only"],
}

MAX_IRRIGATION_ZONES = 5

# Output in the cleaned CSV
CANON_OUTPUT_BASE = ["Time", "AirTemp", "LeafTemp", "RH", "PAR", "LeafWetness"]
IRR_CANONS = [f"Irrigation{i}" for i in range(1, MAX_IRRIGATION_ZONES + 1)]
CANON_OUTPUT_ORDER = CANON_OUTPUT_BASE + IRR_CANONS

# Mapping UI (includes optional Date/TimeOfDay)
CANON_UI_BASE = ["Time", "Date", "TimeOfDay", "AirTemp", "LeafTemp", "RH", "PAR", "LeafWetness"]


# ---------- Automatic data-quality filter limits ---------- 
# Values outside these ranges are replaced with NaN in the cleaned file.

FILTER_REMOVAL_WARNING_PCT = 10.0

FILTER_LIMITS = {
    "AirTemp": {
        "C": (0.0, 50.0),
        "F": (0.0, 120.0),
    },
    "LeafTemp": {
        "C": (0.0, 50.0),
        "F": (0.0, 120.0),
    },
    "RH": (0.0, 100.0),
    "LeafWetness": (0.0, 100.0),
    "PAR": (0.0, None),  # remove negative light values only
}

FILTER_LABELS = {
    "AirTemp": "Air Temperature",
    "LeafTemp": "Leaf Temperature",
    "RH": "Relative Humidity",
    "LeafWetness": "Leaf Wetness",
    "PAR": "Light",
}


def apply_realistic_value_filters(
    df_clean: pd.DataFrame,
    *,
    orig_temp_unit: str = "C",
    show_warnings: bool = False,
) -> pd.DataFrame:
    """
    Replace unrealistic sensor values with NaN so they are ignored in summaries and graphs.
    Does NOT delete rows and does NOT modify the original raw uploaded file.

    show_warnings=True gives a short warning only when >10% of a column is removed.
    """
    df_out = df_clean.copy()

    temp_unit = "F" if str(orig_temp_unit).upper() == "F" else "C"

    for col, limits in FILTER_LIMITS.items():
        if col not in df_out.columns:
            continue

        s = pd.to_numeric(df_out[col], errors="coerce")

        if col in ["AirTemp", "LeafTemp"]:
            low, high = limits[temp_unit]
        else:
            low, high = limits

        valid = pd.Series(True, index=s.index)

        if low is not None:
            valid &= s >= float(low)
        if high is not None:
            valid &= s <= float(high)

        # Only count originally non-missing values.
        original_nonmissing = s.notna()
        removed = original_nonmissing & ~valid

        n_total = int(original_nonmissing.sum())
        n_removed = int(removed.sum())
        pct_removed = (100.0 * n_removed / n_total) if n_total > 0 else 0.0

        df_out[col] = s.mask(removed)

        if show_warnings and pct_removed > FILTER_REMOVAL_WARNING_PCT:
            st.warning(
                f"{pct_removed:.0f}% of {FILTER_LABELS.get(col, col)} values removed - please check data quality"
            )

    return df_out

def regenerate_cleaned_file_with_filters(
    *,
    file_id: int,
    orig_temp_unit: str,
    show_warnings: bool = False,
) -> bool:
    """
    Rebuild cleaned dashboard CSV from saved raw upload,
    apply realistic-value filters, and overwrite only the cleaned CSV.

    Raw uploaded file bytes stay unchanged.
    """
    raw_obj = db.get_raw_file_bytes(int(file_id))
    if raw_obj is None:
        return False

    fcm = db.get_file_column_map(int(file_id))
    if not fcm or not fcm.get("canon_to_raw"):
        return False

    raw_filename = raw_obj["raw_filename"]
    ext = Path(raw_filename).suffix or ".csv"

    df_raw2, _, _, _ = load_table_from_bytes(raw_obj["bytes"], ext)

    canon_to_raw = fcm.get("canon_to_raw") or {}
    raw_to_canon = {
        str(raw): str(canon)
        for canon, raw in canon_to_raw.items()
        if raw
    }

    df_clean2 = build_clean_dataframe(df_raw2, raw_to_canon)

    required_for_dashboard = ["Time", "AirTemp", "RH"]
    missing = [c for c in required_for_dashboard if c not in df_clean2.columns]
    if missing:
        return False

    df_clean2 = apply_realistic_value_filters(
        df_clean2,
        orig_temp_unit=orig_temp_unit,
        show_warnings=show_warnings,
    )

    db.update_file_content(
        int(file_id),
        df_clean2.to_csv(index=False).encode("utf-8"),
    )

    return True

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

def _score_header_candidate(values: list[object], alias_table: dict[str, set]) -> float:
    """
    Score a row as a potential header row.
    Heuristics:
      - more non-empty string-like cells is better
      - more unique cells is better
      - cells that match known alias tokens (time/date/temp/rh/par/etc) boost score
    """
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s or s.lower().startswith("unnamed"):
            continue
        cleaned.append(s)

    if len(cleaned) < 2:
        return -1.0

    uniq_ratio = len(set(cleaned)) / max(len(cleaned), 1)

    # alias hits: any cell that normalizes to something we know
    alias_norms = set().union(*alias_table.values())
    hits = sum(1 for s in cleaned if normalize(s) in alias_norms)

    # small boost if row contains at least one "date" and one "time" token
    norms = [normalize(s) for s in cleaned]
    has_date = any(n in ("date", "day") or "date" in n for n in norms)
    has_time = any(n == "time" or "time" in n for n in norms)

    score = 0.0
    score += len(cleaned) * 1.0
    score += uniq_ratio * 2.0
    score += hits * 5.0
    score += 2.0 if (has_date and has_time) else 0.0

    return score


def detect_header_row_from_preview(df_preview: pd.DataFrame, alias_table: dict[str, set]) -> int:
    """
    Given a preview df read with header=None, return best header row index.
    Defaults to 0 if nothing clearly wins.
    """
    best_row = 0
    best_score = -1.0

    # Only scan first N rows of preview (n_lines=25)
    for r in range(len(df_preview)):
        row_vals = df_preview.iloc[r].tolist()
        s = _score_header_candidate(row_vals, alias_table)
        if s > best_score:
            best_score = s
            best_row = r

    # Require at least a modest score to override row 0
    # (prevents false positives on weird files)
    if best_score < 6.0:
        return 0

    return int(best_row)


def combine_date_time(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    """
    Combine separate Date and Time-of-day columns into a single datetime series.
    Handles:
      - time as strings ("13:05:00")
      - time as Excel fractions of day (numeric)
      - time as datetime64 (we extract time delta)
    """
    d = pd.to_datetime(date_s, errors="coerce")

    # numeric time often means Excel day-fraction
    if pd.api.types.is_numeric_dtype(time_s):
        tnum = pd.to_numeric(time_s, errors="coerce")
        return d.dt.normalize() + pd.to_timedelta(tnum, unit="D")

    t = pd.to_datetime(time_s, errors="coerce")

    # if t is datetime-like, extract time delta within day
    td = t - t.dt.normalize()
    return d.dt.normalize() + td


def load_table_from_bytes(file_bytes: bytes, ext: str) -> tuple[pd.DataFrame, str, str, int]:
    # Load csv or Excel from raw bytes
    ext = ext.lower()
    alias_table = build_alias_table()

    if ext in [".xlsx", ".xls", ".xlsm"]:
        bio = io.BytesIO(file_bytes)

        # preview with no header to detect where headers actually are
        preview = pd.read_excel(bio, header=None, nrows=12)
        header_row = detect_header_row_from_preview(preview, alias_table)

        # re-read full with detected header row
        bio2 = io.BytesIO(file_bytes)
        df = pd.read_excel(bio2, skiprows=header_row, header=0)
        return df, "excel", None, header_row

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None

    for enc in encodings:
        try:
            # Preview using tolerant CSV reader (handles ragged "metadata" rows)
            preview, delim = preview_csv_to_dataframe(file_bytes, enc, n_lines=25)
            header_row = detect_header_row_from_preview(preview, alias_table)

            # Now read the full file, skipping metadata rows above header
            bio2 = io.BytesIO(file_bytes)
            df = pd.read_csv(
                bio2,
                encoding=enc,
                sep=delim,
                skiprows=header_row,
                header=0,
            )
            return df, "csv", enc, header_row

        except Exception as e:
            last_err = e

    raise last_err if last_err is not None else ValueError("Could not read file.")


def build_clean_dataframe(df_raw: pd.DataFrame, raw_to_canon: dict[str, str]) -> pd.DataFrame:
    canon_to_raw: dict[str, str] = {}
    for raw, canon in raw_to_canon.items():
        canon_to_raw.setdefault(canon, raw)

    data = {}

    # -----------------------
    # 1) Build canonical Time
    # -----------------------
    raw_time = canon_to_raw.get("Time")
    raw_date = canon_to_raw.get("Date")
    raw_timeofday = canon_to_raw.get("TimeOfDay")

    time_series = None

    # Prefer combining if Date exists and (TimeOfDay exists OR Time exists)
    if raw_date and raw_date in df_raw.columns and (
        (raw_timeofday and raw_timeofday in df_raw.columns) or (raw_time and raw_time in df_raw.columns)
    ):
        tcol = raw_timeofday if (raw_timeofday and raw_timeofday in df_raw.columns) else raw_time
        time_series = combine_date_time(df_raw[raw_date], df_raw[tcol])

    elif raw_time and raw_time in df_raw.columns:
        time_series = pd.to_datetime(df_raw[raw_time], errors="coerce")

    if time_series is not None:
        data["Time"] = time_series

    # -----------------------
    # 2) Other numeric columns
    # -----------------------
    for canon in CANON_OUTPUT_ORDER:
        if canon == "Time":
            continue

        raw = canon_to_raw.get(canon)
        if raw is None or raw not in df_raw.columns:
            continue

        s = df_raw[raw]
        s = pd.to_numeric(s, errors="coerce")
        data[canon] = s

    if not data:
        return pd.DataFrame()

    df_clean = pd.DataFrame(data)

    if "Time" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["Time"]).sort_values("Time")

    return df_clean.dropna(axis=0, how="all").drop_duplicates()



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


def legend_below(ax, fig, ncol=3, y=-0.50):
    """
    Put the legend below the axes so it doesn't overlap data.
    """
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
    # Make room at the bottom for the legend
    fig.subplots_adjust(bottom=0.28)
    
#No longer used
def plot_separator():
    """
    Add a visual break (space + line) between plots on the Streamlit page.
    """
    #st.markdown("<div style='margin: 1.25rem 0;'></div>", unsafe_allow_html=True)
    #st.divider()
    #st.markdown("<div style='margin: 1.25rem 0;'></div>", unsafe_allow_html=True)

#Naming
def pretty_label(col: str, temp_unit: str) -> str:
    """Readable labels with units (write out words instead of abbreviations)."""
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
    if col == "LeafWetness":
        return "Leaf Wetness"
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

# ==========================================================
# Irrigation helpers
# ==========================================================
_IRR_RE = re.compile(r"^Irrigation(\d+)$", re.IGNORECASE)

def _sorted_irrigation_cols(cols: list[str]) -> list[str]:
    """Sort Irrigation1, Irrigation2, ... numerically."""
    def key(c: str):
        m = _IRR_RE.match(str(c))
        return int(m.group(1)) if m else 999
    return sorted(cols, key=key)

def find_full_days(time_s: pd.Series) -> list[pd.Timestamp]:
    """
    Identify "full days" of data using the same logic style as DLI:
      - >= 20 hours span within the day
      - >= 80% of expected samples (expected based on median logging interval)
    Returns list of midnight timestamps (one per full day), sorted.
    """
    if time_s is None:
        return []
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

def detect_event_times(time_s: pd.Series, on_mask: pd.Series, min_gap_min: float) -> pd.DatetimeIndex:
    """
    Count irrigation events robustly:
      - event candidate = rising edge (off -> on)
      - then enforce min_gap_min between counted events so one long run
        doesn't get counted multiple times across timesteps.
    """
    t = pd.to_datetime(time_s, errors="coerce")
    on = on_mask.fillna(False).astype(bool)

    rising = on & ~on.shift(1, fill_value=False)
    cand = t[rising].dropna().sort_values()
    if cand.empty:
        return pd.DatetimeIndex([])

    if min_gap_min <= 0:
        return pd.DatetimeIndex(cand)

    gap = pd.Timedelta(minutes=float(min_gap_min))
    kept = [cand.iloc[0]]
    last = kept[0]
    for cur in cand.iloc[1:]:
        if (cur - last) >= gap:
            kept.append(cur)
            last = cur

    return pd.DatetimeIndex(kept)

def compute_irrigation_events_per_day(
    df_in: pd.DataFrame,
    irrigation_cols: list[str],
    trigger: float,
    min_gap_min: float,
) -> dict:
    """
    Computes irrigation events per day PER ZONE (column), and returns:

      {
        "events_by_zone": pd.DataFrame indexed by full day (Timestamp),
                          columns are irrigation_cols, values are integer events/day,
        "events_total": pd.Series indexed by full day (sum across zones),
        "full_days": list[Timestamp],
        "day_to_plot": Timestamp|None,
        "binary_by_zone": dict[col -> pd.DataFrame(Time, IrrigationOn)] for day_to_plot
      }

    Event definition:
      - Rising edge (off -> on) after converting numeric signal to ON/OFF using trigger
      - Enforces minimum time between events (min_gap_min) to avoid double counting
        long irrigation runs across multiple log timesteps.
    """
    if "Time" not in df_in.columns or not irrigation_cols:
        return {
            "events_by_zone": None,
            "events_total": None,
            "full_days": [],
            "day_to_plot": None,
            "binary_by_zone": None,
        }

    dfw = df_in[["Time"] + irrigation_cols].copy()
    dfw["Time"] = pd.to_datetime(dfw["Time"], errors="coerce")
    dfw = dfw.dropna(subset=["Time"]).sort_values("Time")
    if dfw.empty:
        return {
            "events_by_zone": None,
            "events_total": None,
            "full_days": [],
            "day_to_plot": None,
            "binary_by_zone": None,
        }

    full_days = find_full_days(dfw["Time"])
    if not full_days:
        return {
            "events_by_zone": None,
            "events_total": None,
            "full_days": [],
            "day_to_plot": None,
            "binary_by_zone": None,
        }

    # -----------------------------
    # 1) Count events/day PER ZONE
    # -----------------------------
    idx_days = pd.to_datetime(full_days)
    events_df = pd.DataFrame(index=idx_days, columns=irrigation_cols, data=0, dtype=int)

    for col in irrigation_cols:
        sig = pd.to_numeric(dfw[col], errors="coerce").fillna(0.0)
        on = sig >= float(trigger)

        # Rising-edge + min-gap event times for this zone
        ev_times = detect_event_times(dfw["Time"], on, float(min_gap_min))

        if len(ev_times) == 0:
            continue

        ev_dates = pd.Series(pd.to_datetime(ev_times)).dt.date
        for d in idx_days:
            events_df.at[d, col] = int((ev_dates == d.date()).sum())

    events_total = events_df.sum(axis=1)
    events_total.name = "IrrigationEventsPerDay_Total"

    # -----------------------------------------------
    # 2) Build 0/1 ON/OFF series per zone for 1 day
    # -----------------------------------------------
    day_to_plot = idx_days[-1]
    day_mask = dfw["Time"].dt.date == day_to_plot.date()

    binary_by_zone = {}
    for col in irrigation_cols:
        sig = pd.to_numeric(dfw.loc[day_mask, col], errors="coerce").fillna(0.0)
        on = (sig >= float(trigger)).astype(int)

        binary_by_zone[col] = pd.DataFrame(
            {"Time": dfw.loc[day_mask, "Time"].values, "IrrigationOn": on.values}
        )

    return {
        "events_by_zone": events_df,
        "events_total": events_total,
        "full_days": full_days,
        "day_to_plot": day_to_plot,
        "binary_by_zone": binary_by_zone,
    }

#Helpers for updated editable table
IGNORE_RAW = "(None)"
IRRIGATION_EVENTS_ROW_LABEL = "Irrigation Events per Day (#)"

SUMMARY_CANON_ROWS = [
    ("AirTemp", "Air Temperature"),
    ("LeafTemp", "Leaf Temperature"),
    ("RH", "Relative Humidity"),
    ("VPD", "VPD"),
    ("PAR", "Light Intensity"),
    ("DLI", "DLI"),
    ("LeafWetness", "Leaf Wetness"),
    # irrigation rows stay as-is in existing summary logic
]

def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        v = float(x)
        # NaN check
        if v != v:
            return None
        return v
    except Exception:
        return None


def _canon_to_raw_from_fcm(fcm_row, fallback_raw_cols):
    canon_to_raw = {canon: None for canon in CANON_OUTPUT_ORDER}
    raw_cols = list(fallback_raw_cols)

    if fcm_row:
        if fcm_row.get("canon_to_raw"):
            # stored JSONB already deserialized by RealDictCursor
            for k, v in (fcm_row.get("canon_to_raw") or {}).items():
                canon_to_raw[str(k)] = v if v else None
        if fcm_row.get("raw_columns"):
            raw_cols = [str(c) for c in (fcm_row.get("raw_columns") or [])] or raw_cols

    return canon_to_raw, raw_cols


def _invert_canon_to_raw(canon_to_raw: dict) -> dict:
    """{canon: raw} -> {raw: canon} dropping Nones"""
    out = {}
    for canon, raw in (canon_to_raw or {}).items():
        if raw:
            out[str(raw)] = str(canon)
    return out

# --- Small status badges (pills) used under metrics ---
st.markdown(
    """
    <style>
    .metric-badge{
        display:inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.35rem;
    }
    .badge-good{ background:#E9F7EF; color:#1E7E34; }  /* green */
    .badge-high{ background:#FDECEC; color:#B02A37; }  /* red */
    .badge-low{  background:#E7F1FF; color:#0B5ED7; }  /* blue */
    .badge-na{   background:#F1F3F5; color:#495057; }  /* gray */
    </style>
    """,
    unsafe_allow_html=True
)

def badge_html(text: str, cls: str) -> str:
    return f"<div class='metric-badge {cls}'>{text}</div>"

#Styling for summary statistics table - edit column width
st.markdown("""
    <style>
    /* Target pandas styler tables */
    table { 
    width: 100% !important;
    table-layout: fixed !important;   /* <-- key: obey widths */
    }

    /* First column (row labels) narrower */
    table th.row_heading, table td.row_heading,
    table th:first-child, table td:first-child {
    width: 28% !important;
    white-space: normal;             /* allow wrap */
    }

    /* Make numeric columns equal */
    table th.col_heading, table td {
    text-align: center;
    }

    /* If your table is exactly 6 columns (Label + Low & High Targets + Min + Avg + Max) */
    table th:nth-child(2), table td:nth-child(2),
    table th:nth-child(3), table td:nth-child(3),
    table th:nth-child(4), table td:nth-child(4),
    table th:nth-child(5), table td:nth-child(5) {
    width: 14% !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Scope only to the summary editor (its key becomes a CSS class) */
    [class*="st-key-home_summary_editor_"] [data-testid="stDataEditor"] [role="columnheader"] {
        justify-content: center !important;
        text-align: center !important;
    }

    [class*="st-key-home_summary_editor_"] [data-testid="stDataEditor"] [role="columnheader"] * {
        justify-content: center !important;
        text-align: center !important;
    }

    /* Grid cells */
    [class*="st-key-home_summary_editor_"] [data-testid="stDataEditor"] [role="gridcell"] {
        text-align: center !important;
    }

    [class*="st-key-home_summary_editor_"] [data-testid="stDataEditor"] [role="gridcell"] > div {
        width: 100% !important;
        justify-content: center !important;
        text-align: center !important;
    }

    /* Editable number/text inputs */
    [class*="st-key-home_summary_editor_"] [data-testid="stDataEditor"] input {
        text-align: center !important;
    }

    /* Selectbox display */
    [class*="st-key-home_summary_editor_"] [data-testid="stDataEditor"] [data-baseweb="select"] > div,
    [class*="st-key-home_summary_editor_"] [data-testid="stDataEditor"] [data-baseweb="select"] span {
        justify-content: center !important;
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------- Main content ----------

user = auth.current_user()
#st.title("🌿 EnDash")
#st.divider()

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

# # Top-row navigation & actions
# col1, col2, col3 = st.columns(3, gap="medium")

# with col1:
#     if st.button("📂 Data File Settings", width="stretch"):
#         st.switch_page("pages/1_Upload.py")

# with col2:
#     if st.button("⚙️ Climate Units & Setpoints", width="stretch"):
#         st.switch_page("pages/2_Settings.py")

# with col3:
#     download_slot = st.empty()

# ----- Quick Upload panel (always visible) -----
logo_col, left_col, mid_col = st.columns([0.8, 1.3, 1.3], gap="medium")

with logo_col:
    st.image("assets/EnDash_Logo_V3.png", use_container_width=True)

with left_col:
    #st.markdown("### New File Upload")
    #st.subheader("New File Upload", help="Upload a file to add it to your dashboard.")
    #st.caption()

    #Functions added to accept variable row of column headings and separate Date/Time
    def detect_delimiter_from_lines(lines: list[str]) -> str:
        """
        Pick a delimiter by counting occurrences across the first few non-empty lines.
        Works well for comma, tab, semicolon, pipe.
        """
        candidates = [",", "\t", ";", "|"]
        best = ","
        best_score = -1
        for d in candidates:
            score = sum(line.count(d) for line in lines if line.strip())
            if score > best_score:
                best_score = score
                best = d
        return best


    def preview_csv_to_dataframe(file_bytes: bytes, encoding: str, n_lines: int = 25) -> tuple[pd.DataFrame, str]:
        """
        Read first n_lines using csv.reader (tolerant of ragged rows), return a DataFrame + detected delimiter.
        """
        text = file_bytes.decode(encoding, errors="replace")
        lines = text.splitlines()[:n_lines]

        # If file is mostly empty, return empty df
        if not any(l.strip() for l in lines):
            return pd.DataFrame(), ","

        delim = detect_delimiter_from_lines(lines)

        reader = csv.reader(StringIO("\n".join(lines)), delimiter=delim)
        rows = [r for r in reader]

        # Normalize to rectangular table (pad shorter rows)
        max_cols = max((len(r) for r in rows), default=0)
        rows = [r + [None] * (max_cols - len(r)) for r in rows]

        df_preview = pd.DataFrame(rows)
        return df_preview, delim

    quick_file = st.file_uploader(
        "Upload a new file:",
        type=["csv", "xlsx", "xls", "xlsm"],
        key="quick_upload_file",
        #help="Drop a data file here to add it to your dashboard.",
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
                df_raw, file_type, encoding_used, header_row = load_table_from_bytes(file_bytes_raw, ext)
                if header_row > 0:
                    st.info(f"Detected headers on row {header_row+1} (skipped {header_row} row(s) above).")

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
                settings_for_filter = dict(db.get_or_create_settings(user["id"]) or {})
                orig_temp_unit_for_filter = settings_for_filter.get("orig_temp_unit", "C")

                df_clean = apply_realistic_value_filters(
                    df_clean,
                    orig_temp_unit=orig_temp_unit_for_filter,
                    show_warnings=True,
                )

                # Build stored filename: "<OriginalFilename>_<Username>.csv"
                uname = username_slug(user)

                orig_stem = Path(original_name).stem
                orig_stem = re.sub(r"\s+", "_", orig_stem)
                orig_stem = re.sub(r"[^A-Za-z0-9_-]+", "", orig_stem)
                orig_stem = re.sub(r"_+", "_", orig_stem).strip("_") or "file"

                stored_filename = f"{orig_stem}_{uname}.csv"

                cleaned_bytes = df_clean.to_csv(index=False).encode("utf-8")
                file_db_id = db.add_file_record(
                    user["id"],
                    stored_filename,
                    cleaned_bytes,
                    raw_filename=original_name,
                    raw_bytes=file_bytes_raw,
                )

                filter_key = f"{int(file_db_id)}_{orig_temp_unit_for_filter}"

                if "filtered_file_unit_keys" not in st.session_state:
                    st.session_state["filtered_file_unit_keys"] = set()

                st.session_state["filtered_file_unit_keys"].add(filter_key)

                # Save mapping metadata so it can be reviewed/edited on the Upload page later
                try:
                    canon_to_raw = {canon: None for canon in CANON_OUTPUT_ORDER}
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

                # # 3) Build cleaned dataframe
                # df_clean = build_clean_dataframe(df_raw, auto_mapping)

                # required_for_dashboard = ["Time", "AirTemp", "RH"]
                # missing_for_dashboard = [c for c in required_for_dashboard if c not in df_clean.columns]
                # if missing_for_dashboard:
                #     raise ValueError("Missing required columns for dashboard: " + ", ".join(missing_for_dashboard))

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

with mid_col:
    # ----- File selection (Next to Quick Upload) -----
    #st.subheader("Current File Selection", help="Select a stored file to view.")

    files = db.list_user_files(user["id"])
    if not files:
        st.info("No cleaned files found yet. Upload a file above to get started.")
        st.stop()

    # Build stable option list using file IDs (NOT label strings)
    id_to_rec = {int(rec["id"]): rec for rec in files}
    file_ids = list(id_to_rec.keys())  # list is already most-recent first from DB

    def _home_file_label(fid: int) -> str:
        r = id_to_rec[int(fid)]
        return f"{r['filename']} ({r['uploaded_at']})"

    # DB-backed default (same pattern as Upload page)
    last_home_id = db.get_last_home_file_id(user["id"])
    default_id = last_home_id if (last_home_id in id_to_rec) else file_ids[0]

    # Initialize session state ONCE (do not overwrite on every rerun)
    if "home_selected_file_id" not in st.session_state:
        st.session_state.home_selected_file_id = default_id

    # If the selected ID no longer exists (deleted), fall back safely
    if int(st.session_state.home_selected_file_id) not in id_to_rec:
        st.session_state.home_selected_file_id = default_id

    selected_file_id = st.selectbox(
        "Select a stored file to view:",
        file_ids,
        format_func=_home_file_label,
        key="home_selected_file_id",
    )

    # Persist selection like Upload page does (DB + session)
    if st.session_state.get("home_selected_file_id_last_saved") != int(selected_file_id):
        db.set_last_home_file_id(user["id"], int(selected_file_id))
        st.session_state.home_selected_file_id_last_saved = int(selected_file_id)

    rec = id_to_rec[int(selected_file_id)]
    st.session_state["selected_file_id"] = int(selected_file_id) 

    download_slot = st.empty()

st.markdown("---")

if rec is None:
    st.stop()

# ----- Load per-user settings BEFORE selected file is read -----
settings_row = db.get_or_create_settings(user["id"])
settings = dict(settings_row) if settings_row is not None else {}

orig_temp_unit = settings.get("orig_temp_unit", "C")

# ----- Filter existing selected file only once per file/unit per session -----
if "filtered_file_unit_keys" not in st.session_state:
    st.session_state["filtered_file_unit_keys"] = set()

selected_filter_key = f"{int(rec['id'])}_{orig_temp_unit}"

if selected_filter_key not in st.session_state["filtered_file_unit_keys"]:
    regenerated = regenerate_cleaned_file_with_filters(
        file_id=int(rec["id"]),
        orig_temp_unit=orig_temp_unit,
        show_warnings=True,
    )

    # Mark it as checked either way so an older file without raw bytes
    # does not keep retrying on every rerun.
    st.session_state["filtered_file_unit_keys"].add(selected_filter_key)

    if not regenerated:
        st.caption("Filter update skipped.")


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


# ----- Home-only Units section in sidebar -----
units_settings = settings

current_orig_temp_unit = units_settings.get("orig_temp_unit", "C")
current_orig_light_unit = units_settings.get("orig_light_unit", "PPFD")
current_leaf_wetness_unit = units_settings.get("leaf_wetness_unit", "Percent")
current_dashboard_temp_unit = units_settings.get("temp_unit", "F")

TEMP_UNIT_OPTIONS = {
    "Celsius (°C)": "C",
    "Fahrenheit (°F)": "F",
}

LIGHT_UNIT_OPTIONS = {
    "PPFD (µmol m⁻² s⁻¹)": "PPFD",
    "Lux": "LUX",
    "Kilolux (klux)": "KLUX",
    "Footcandles (fc)": "FC",
    "W m⁻²": "W_M2",
}

LEAF_WETNESS_OPTIONS = ["Percent", "Volts", "milliVolts"]

def _temp_unit_index(current_code: str) -> int:
    labels = list(TEMP_UNIT_OPTIONS.keys())
    for i, lbl in enumerate(labels):
        if TEMP_UNIT_OPTIONS[lbl] == current_code:
            return i
    return 0

def _light_unit_index(current_code: str) -> int:
    labels = list(LIGHT_UNIT_OPTIONS.keys())
    for i, lbl in enumerate(labels):
        if LIGHT_UNIT_OPTIONS[lbl] == current_code:
            return i
    return 0

if current_leaf_wetness_unit not in LEAF_WETNESS_OPTIONS:
    current_leaf_wetness_unit = "Percent"

with st.sidebar:
    st.divider()
    st.subheader("Units")

    with st.form("home_units_form_sidebar", clear_on_submit=False):
        orig_temp_choice = st.selectbox(
            "Data File Temperature",
            options=list(TEMP_UNIT_OPTIONS.keys()),
            index=_temp_unit_index(current_orig_temp_unit),
        )

        orig_light_choice = st.selectbox(
            "Data File Light",
            options=list(LIGHT_UNIT_OPTIONS.keys()),
            index=_light_unit_index(current_orig_light_unit),
        )

        dashboard_temp_choice = st.selectbox(
            "Dashboard Temperature",
            options=list(TEMP_UNIT_OPTIONS.keys()),
            index=_temp_unit_index(current_dashboard_temp_unit),
        )

        save_units = st.form_submit_button("Save Units", use_container_width=True)

    if save_units:
        new_orig_temp_unit = TEMP_UNIT_OPTIONS[orig_temp_choice]
        new_orig_light_unit = LIGHT_UNIT_OPTIONS[orig_light_choice]
        new_dashboard_temp_unit = TEMP_UNIT_OPTIONS[dashboard_temp_choice]
        db.update_settings(
            user["id"],
            orig_temp_unit=new_orig_temp_unit,
            orig_light_unit=new_orig_light_unit,
            temp_unit=new_dashboard_temp_unit,
            target_low=float(units_settings.get("target_low", 65.0)),
            target_high=float(units_settings.get("target_high", 80.0)),
            target_rh_low=float(units_settings.get("target_rh_low", 70.0)),
            target_rh_high=float(units_settings.get("target_rh_high", 95.0)),
            target_ppfd=float(units_settings.get("target_ppfd", 150.0)),
            target_dli_low=float(units_settings.get("target_dli_low", 8.0)),
            target_dli_high=float(units_settings.get("target_dli_high", 12.0)),
            target_vpd_low=float(units_settings.get("target_vpd_low", 0.2)),
            target_vpd_high=float(units_settings.get("target_vpd_high", 0.8)),
            irrigation_trigger=float(units_settings.get("irrigation_trigger", 1.0)),
            irrigation_min_interval_min=float(units_settings.get("irrigation_min_interval_min", 7.0)),
            leaf_wetness_unit=units_settings.get("leaf_wetness_unit", "Percent"),
            irrigation_sensitivity_pct=float(units_settings.get("irrigation_sensitivity_pct", 3.0)),
            leaf_wetness_min_interval_min=float(units_settings.get("leaf_wetness_min_interval_min", 7.0)),
            water_applied_per_event_ml_m2=float(units_settings.get("water_applied_per_event_ml_m2", 10.0)),
        )
        new_orig_temp_unit = TEMP_UNIT_OPTIONS[orig_temp_choice]

        regenerated = regenerate_cleaned_file_with_filters(
            file_id=int(selected_file_id),
            orig_temp_unit=new_orig_temp_unit,
            show_warnings=True,
        )

        if "filtered_file_unit_keys" not in st.session_state:
            st.session_state["filtered_file_unit_keys"] = set()

        st.session_state["filtered_file_unit_keys"].add(
            f"{int(selected_file_id)}_{new_orig_temp_unit}"
        )

        if not regenerated:
            st.warning("Units saved, but file could not be regenerated.")
        st.rerun()


# ----- Load saved mapping for THIS file (per user per file) -----
fcm = db.get_file_column_map(int(rec["id"]))
canon_to_raw_saved, raw_cols_for_dropdown = _canon_to_raw_from_fcm(fcm, fallback_raw_cols=list(df.columns))

# Units
temp_unit = settings.get("temp_unit", "F")  # display unit 'F' or 'C'
orig_temp_unit = settings.get("orig_temp_unit", None)  # 'C' or 'F'
orig_light_unit = settings.get("orig_light_unit", "PPFD")

# Targets / Default Setpoints
target_temp_low = float(settings.get("target_low", 65.0))
target_temp_high = float(settings.get("target_high", 85.0))

target_rh_low = float(settings.get("target_rh_low", 50.0))
target_rh_high = float(settings.get("target_rh_high", 90.0))

target_ppfd = float(settings.get("target_ppfd", 750.0))
target_dli_low = float(settings.get("target_dli_low", 8.0))
target_dli_high = float(settings.get("target_dli_high", 12.0))

target_vpd_low = float(settings.get("target_vpd_low", 0.2))
target_vpd_high = float(settings.get("target_vpd_high", 1.5))

irrigation_trigger = float(settings.get("irrigation_trigger", 1.0)) #ON if value >= trigger
irrigation_min_interval_min = float(settings.get("irrigation_min_interval_min", 7.0))   #minutes
water_applied_per_event_ml_m2 = float(settings.get("water_applied_per_event_ml_m2", 10))

leaf_wetness_unit = settings.get("leaf_wetness_unit", "Percent")
irrigation_sensitivity_pct = float(settings.get("irrigation_sensitivity_pct", 3.0))
leaf_wetness_min_interval_min = float(settings.get("leaf_wetness_min_interval_min", 7.0))

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

# Compute DLI only when PAR is PPFD
daily_dli_series = None
if orig_light_unit == "PPFD" and "Time" in df_display.columns and "PAR" in df_display.columns:
    daily_dli_series = compute_daily_dli(df_display[["Time", "PAR"]])

# =========================
# Dashboard Summary section
# =========================
#st.subheader("Dashboard Summary", help="Upload a new file or select one from above to view your dashboard. To edit column selections, use the Data File Settings page. To edit units and target setpoints, use the Climate Units and Setpoints page.")
#st.caption(f"Showing selected file: `{rec['filename']}` (uploaded {rec['uploaded_at']}).")

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
    st.subheader(
        f"Data Summary from "
        f"**{start_time.strftime('%Y-%m-%d %H:%M')}** "
        f"to **{end_time.strftime('%Y-%m-%d %H:%M')}** "
        f" - Interval: **{interval_str}**",
        help="Upload a new file or select a stored file to view your data summary."
    )
else:
    st.caption("Time information is not available in this file.")


pdf_summary_title = None

if start_time is not None and end_time is not None:
    pdf_summary_title = (
        f"Data Summary from {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"to {end_time.strftime('%Y-%m-%d %H:%M:%S')} - Interval: {interval_str}"
    )

# # -----------------------------
# # Summary metrics (3 columns)
# # -----------------------------
# metric_cols = st.columns(3, gap="medium")

# # --- Air Temperature ---
# temp_badge = badge_html("n/a", "badge-na")
# temp_within_pct = None

# if air_disp is not None and air_disp.notna().any():
#     air_mean = float(air_disp.mean(skipna=True))

#     temp_series = air_disp.dropna()
#     if len(temp_series) > 0:
#         within_mask = (temp_series >= target_temp_low) & (temp_series <= target_temp_high)
#         temp_within_pct = 100.0 * float(within_mask.mean())

#     # Determine mean position relative to band (for label text + color)
#     if air_mean < target_temp_low:
#         state_txt = "Below target band"
#         cls = "badge-low"
#     elif air_mean > target_temp_high:
#         state_txt = "Above target band"
#         cls = "badge-high"
#     else:
#         state_txt = "Within target band"
#         cls = "badge-good"

#     pct_txt = "-" if temp_within_pct is None else f"{temp_within_pct:.0f}% within range"
#     temp_badge = badge_html(f"{state_txt} · {pct_txt}", cls)

#     metric_cols[0].metric(
#         label=f"Average Air Temperature ({'°F' if temp_unit == 'F' else '°C'})",
#         value=f"{air_mean:.1f}",
#     )
#     metric_cols[0].markdown(temp_badge, unsafe_allow_html=True)
# else:
#     metric_cols[0].metric(
#         label=f"Average Air Temperature ({'°F' if temp_unit == 'F' else '°C'})",
#         value="—",
#     )
#     metric_cols[0].markdown(temp_badge, unsafe_allow_html=True)

# # --- Relative Humidity ---
# rh_badge = badge_html("n/a", "badge-na")
# rh_within_pct = None

# if rh is not None and rh.notna().any():
#     rh_mean = float(rh.mean(skipna=True))

#     rh_series = rh.dropna()
#     if len(rh_series) > 0:
#         rh_within_mask = (rh_series >= target_rh_low) & (rh_series <= target_rh_high)
#         rh_within_pct = 100.0 * float(rh_within_mask.mean())

#     if rh_mean < target_rh_low:
#         state_txt = "Below target band"
#         cls = "badge-low"
#     elif rh_mean > target_rh_high:
#         state_txt = "Above target band"
#         cls = "badge-high"
#     else:
#         state_txt = "Within target band"
#         cls = "badge-good"

#     pct_txt = "-" if rh_within_pct is None else f"{rh_within_pct:.0f}% within range"
#     rh_badge = badge_html(f"{state_txt} · {pct_txt}", cls)

#     metric_cols[1].metric(
#         label="Average Relative Humidity (%)",
#         value=f"{rh_mean:.0f}",
#     )
#     metric_cols[1].markdown(rh_badge, unsafe_allow_html=True)
# else:
#     metric_cols[1].metric(
#         label="Average Relative Humidity (%)",
#         value="—",
#     )
#     metric_cols[1].markdown(rh_badge, unsafe_allow_html=True)

# # --- DLI (no band; above/below setpoint only) ---
# dli_badge = badge_html("n/a", "badge-na")

# if daily_dli_series is not None and not daily_dli_series.empty:
#     dli_mean = float(daily_dli_series.mean())

#     pct_days_below = 100.0 * float((daily_dli_series < target_dli_low).mean())
#     pct_days_within = 100.0 * float(((daily_dli_series >= target_dli_low) & (daily_dli_series <= target_dli_high)).mean())
#     pct_days_above = 100.0 * float((daily_dli_series > target_dli_high).mean())

#     # Badge state based on MEAN (simple + intuitive)
#     if dli_mean < target_dli_low:
#         state_txt = "Below band"
#         cls = "badge-high"      # red
#     elif dli_mean > target_dli_high:
#         state_txt = "Above band"
#         cls = "badge-high"      # red (or create a separate warning class if you want)
#     else:
#         state_txt = "Within band"
#         cls = "badge-good"      # green

#     dli_badge = badge_html(f"{state_txt} · {pct_days_within:.0f}% days within target", cls)

#     metric_cols[2].metric(
#         label="Average DLI (mol m⁻² d⁻¹)",
#         value=f"{dli_mean:.1f}",
#     )
#     metric_cols[2].markdown(dli_badge, unsafe_allow_html=True)
# else:
#     metric_cols[2].metric(
#         label="Average DLI (mol m⁻² d⁻¹)",
#         value="—",
#     )
#     metric_cols[2].markdown(dli_badge, unsafe_allow_html=True)

# Optional single highlight message 
#if temp_within_pct is not None:
#    if temp_within_pct < 50:
#        st.error(f"About **{temp_within_pct:.0f}%** of air temperature readings were within your target band.")
#    elif temp_within_pct < 80:
#        st.warning(f"About **{temp_within_pct:.0f}%** of air temperature readings were within your target band.")
#    else:
#        st.success(f"About **{temp_within_pct:.0f}%** of air temperature readings were within your target band.")


# =========================
# Summary Statistics
# =========================
#st.subheader("Summary Statistics")

# --- Identify irrigation canonical columns (Irrigation1..IrrigationN) ---
irrigation_cols = _sorted_irrigation_cols(
    [c for c in df_display.columns if _IRR_RE.match(str(c))] + ([c for c in df_display.columns if str(c).lower() == "irrigation"])
)
# Remove duplicates if both "Irrigation" and "Irrigation1" exist, etc.
irrigation_cols = list(dict.fromkeys(irrigation_cols))

# --- Compute irrigation events/day (uses peak/rising-edge + min-gap rule) ---
irrig_stats = compute_irrigation_events_per_day(
    df_display,
    irrigation_cols=irrigation_cols,
    trigger=irrigation_trigger,
    min_gap_min=irrigation_min_interval_min,
)

events_by_zone = irrig_stats.get("events_by_zone", None)   # DataFrame (days x zones)
events_total = irrig_stats.get("events_total", None)       # Series (days)


# Build the summary table from numeric columns EXCLUDING irrigation raw signals
numeric_cols = df_display.select_dtypes(include="number").columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in irrigation_cols]
# do not show the 0/1 leaf-wetness event marker column in Summary Statistics
numeric_cols = [c for c in numeric_cols if c != "LeafWetness"]
numeric_cols = [c for c in numeric_cols if c != "IrrigationEvents_LeafWetness"]

summary = None
summary_display = None
summary_numeric = None

if numeric_cols:
    summary = df_display[numeric_cols].agg(["min", "mean", "max"]).transpose()
    summary.rename(columns={"min": "Min", "mean": "Average", "max": "Max"}, inplace=True)
    summary.index = [pretty_label(c, temp_unit) for c in summary.index]

    # ---- Add Target columns (numeric; NaN means show "-") ----
    summary["Low Target"] = np.nan
    summary["High Target"] = np.nan

    def _set_targets(label_contains: str, low, high):
        mask = summary.index.to_series().astype(str).str.contains(label_contains, regex=False)
        if low is not None:
            summary.loc[mask, "Low Target"] = float(low)
        if high is not None:
            summary.loc[mask, "High Target"] = float(high)

    # Air + Leaf temperature use same band
    _set_targets("Air Temperature", target_temp_low, target_temp_high)
    _set_targets("Leaf Temperature", target_temp_low, target_temp_high)

    # RH band
    _set_targets("Relative Humidity", target_rh_low, target_rh_high)

    # VPD band
    _set_targets("Vapor Pressure Deficit", target_vpd_low, target_vpd_high)

    # Light Intensity (PPFD): only a single target -> put it in High Target, Low Target stays "-"
    _set_targets("Light Intensity", None, target_ppfd)


    # ---- Add DLI row (if available) ----
    if daily_dli_series is not None and not daily_dli_series.empty:
        dli_row = pd.DataFrame(
            {"Min": [daily_dli_series.min()], "Average": [daily_dli_series.mean()], "Max": [daily_dli_series.max()]},
            index=["Daily Light Integral (mol m⁻² d⁻¹)"],
        )
        summary = pd.concat([summary, dli_row], axis=0)
       
        # DLI band Target column (now that the DLI row exists)
        _set_targets("Daily Light Integral", target_dli_low, target_dli_high)

        # ---- Add ONE irrigation events/day row (full days only) ----
        if events_total is not None and not events_total.empty:
            s = events_total.astype(float)

            irrig_row = pd.DataFrame(
                {
                    "Min": [float(s.min())],
                    "Average": [float(s.mean())],
                    "Max": [float(s.max())],
                },
                index=[IRRIGATION_EVENTS_ROW_LABEL],
            )
            summary = pd.concat([summary, irrig_row], axis=0)

    # ---- Add Water Applied per Day rows  ----
    if water_applied_per_event_ml_m2 > 0:
        # Based on irrigation events/day 
        if events_total is not None and not events_total.empty:
            wapd = events_total.astype(float) * float(water_applied_per_event_ml_m2)
            wapd_row = pd.DataFrame(
                {"Min": [float(wapd.min())], "Average": [float(wapd.mean())], "Max": [float(wapd.max())]},
                index=["Water Applied per Day (mL m⁻² d⁻¹)"],
            )
            summary = pd.concat([summary, wapd_row], axis=0)

    def label_with_icon(label: str) -> str:
        """
        Add an emoji icon prefix based on the row label text.
        Order matters (more specific rules first).
        """
        rules = [
            ("Events per Day", "🚿"),            # irrigation events/day
            ("Water Applied", "💧"),                   # water applied/day
            ("Daily Light Integral", "🌤️"),      # DLI
            ("Light Intensity", "💡"),           # PPFD
            ("Vapor Pressure Deficit", "🍃"),    # VPD
            ("Relative Humidity", "💦"),         # RH
            ("Leaf Temperature", "🍂"),
            ("Temperature", "🌡️"),              # temps
            ("Leaf Wetness", "🌿"),              # leaf wetness
        ]
        for key, emoji in rules:
            if key in label:
                return f"{emoji} {label}"
        return label

    # ---- Add emoji icons to summary row labels (index) ----
    summary.index = [label_with_icon(str(i)) for i in summary.index]

    # Column order for display table/editor
    summary = summary[["Low Target", "High Target", "Min", "Average", "Max"]]

    def _visible_irrigation_canons() -> list[str]:
        """
        Mirror the Upload-page behavior as closely as possible:
        Always show at least Irrigation1.
        """
        found = []
        for i in range(1, MAX_IRRIGATION_ZONES + 1):
            canon = f"Irrigation{i}"
            if (
                canon_to_raw_saved.get(canon)
                or canon in df.columns
                or canon in df_display.columns
            ):
                found.append(canon)

        return found or ["Irrigation1"]

    visible_irrig_canons = _visible_irrigation_canons()

    # Pick whichever VPD label exists in this file (leaf preferred if present)
    vpd_candidates = [idx for idx in summary.index if "Vapor Pressure Deficit" in str(idx)]
    vpd_label = vpd_candidates[0] if vpd_candidates else "🍃 Leaf Vapor Pressure Deficit (kPa)"

    temp_symbol = "°F" if temp_unit == "F" else "°C"

    desired_rows = [
        f"🌡️ Air Temperature ({temp_symbol})",
        f"🍂 Leaf Temperature ({temp_symbol})",
        "💦 Relative Humidity (%)",
        vpd_label,
        "💡 Light Intensity (PPFD - µmol m⁻² s⁻¹)",
        f"🚿 {IRRIGATION_EVENTS_ROW_LABEL}",
        "🌤️ Daily Light Integral (mol m⁻² d⁻¹)",
        "💧 Water Applied per Day (mL m⁻² d⁻¹)",
    ]


    # Reindex to force all rows to exist; missing ones become NaN -> later display as "-"
    summary = summary.reindex(desired_rows)


    # --- Display formatting: per-row decimals + PPFD Min/Average as "-" ---
    ppfd_label_base = "Light Intensity (PPFD - µmol m⁻² s⁻¹)"
    ppfd_label = label_with_icon(ppfd_label_base)

    def row_format_spec(row_label: str) -> str:
        if "Events per Day" in row_label:
            return "{:.0f}"
        if "Water Applied" in row_label:
            return "{:.0f}"
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
        if "Leaf Wetness" in row_label:
            return "{:.2f}"
        return "{:.1f}"

    def fmt_cell(val, fmt: str) -> str:
        if pd.isna(val):
            return "-"
        try:
            return fmt.format(float(val))
        except Exception:
            return str(val)

    summary_display = summary.copy().astype("object")
    summary_numeric = summary.copy()

    # Force PPFD Min/Average to be "-"
    if ppfd_label in summary_display.index:
        summary_display.loc[ppfd_label, "Min"] = np.nan
        summary_display.loc[ppfd_label, "Average"] = np.nan

    for idx in summary_display.index:
        fmt = row_format_spec(str(idx))
        for col in summary_display.columns:
            if idx == ppfd_label and col in ["Min", "Average"]:
                summary_display.at[idx, col] = "-"
            else:
                summary_display.at[idx, col] = fmt_cell(summary_display.at[idx, col], fmt)

    # --- Cell coloring based on targets (irrigation stays black) ---
    def build_style_df(df_disp: pd.DataFrame, df_num: pd.DataFrame) -> pd.DataFrame:
        style = pd.DataFrame("", index=df_disp.index, columns=df_disp.columns)

        def set_color(i, j, color: str):
            style.at[i, j] = f"color: {color};"

        for row_label in df_disp.index:
            for col in df_disp.columns:
                try:
                    val = df_num.loc[row_label, col]
                except Exception:
                    val = np.nan

                # Don't color the Target columns (keep them neutral)
                if col in ["Data Column", "Low Target", "High Target"]:
                    set_color(row_label, col, "black")
                    continue

                if pd.isna(val):
                    set_color(row_label, col, "black")
                    continue

                # PPFD & DLI
                if row_label == ppfd_label:
                    set_color(row_label, col, "green" if float(val) < float(target_ppfd) else "red")
                    continue

                if "Daily Light Integral" in str(row_label):
                    v = float(val)
                    if v < float(target_dli_low):
                        set_color(row_label, col, "blue")
                    elif v > float(target_dli_high):
                        set_color(row_label, col, "red")
                    else:
                        set_color(row_label, col, "green")
                    continue

                # Temperature rows
                if "Temperature" in str(row_label):
                    if float(val) < float(target_temp_low):
                        set_color(row_label, col, "blue")
                    elif float(val) > float(target_temp_high):
                        set_color(row_label, col, "red")
                    else:
                        set_color(row_label, col, "green")
                    continue

                # RH row
                if "Relative Humidity" in str(row_label):
                    if float(val) < float(target_rh_low):
                        set_color(row_label, col, "blue")
                    elif float(val) > float(target_rh_high):
                        set_color(row_label, col, "red")
                    else:
                        set_color(row_label, col, "green")
                    continue

                # VPD rows
                if "Vapor Pressure Deficit" in str(row_label):
                    if float(val) < float(target_vpd_low):
                        set_color(row_label, col, "blue")
                    elif float(val) > float(target_vpd_high):
                        set_color(row_label, col, "red")
                    else:
                        set_color(row_label, col, "green")
                    continue

                # Everything else (including irrigation events/day): black
                set_color(row_label, col, "black")

        return style

    # ---------- Editable summary table (replace old HTML styler table) ----------
    LOCKED_DATA_COLUMN = "-"
    UNMAPPED_DATA_COLUMN = "Blank - Click to Select Data"

    def _build_data_column_tokens(raw_cols: list[str]):
        """
        Build 1-based tokens for raw columns:
        token "1" -> first raw column
        token "2" -> second raw column
        """
        token_to_raw = {str(i + 1): str(raw) for i, raw in enumerate(raw_cols)}
        raw_to_token = {raw: token for token, raw in token_to_raw.items()}
        token_to_label = {token: f"[{token}] {raw}" for token, raw in token_to_raw.items()}
        return token_to_raw, raw_to_token, token_to_label

    token_to_raw, raw_to_token, token_to_label = _build_data_column_tokens(raw_cols_for_dropdown)

    def _format_data_column_option(val):
        """
        How Data Column values appear in the dropdown/editor.
        Keep sentinel values as-is.
        """
        if val in (None, "", LOCKED_DATA_COLUMN, UNMAPPED_DATA_COLUMN):
            return val
        return token_to_label.get(str(val), str(val))

    def _summary_row_to_canon(row_label: str) -> str | None:
        s = str(row_label)

        if "Air Temperature" in s:
            return "AirTemp"
        if "Leaf Temperature" in s:
            return "LeafTemp"
        if "Relative Humidity" in s:
            return "RH"
        if "Light Intensity" in s:
            return "PAR"
        if IRRIGATION_EVENTS_ROW_LABEL in s:
            return "Irrigation1"

        # derived-only rows: no direct mapping
        if "Vapor Pressure Deficit" in s:
            return None
        if "Daily Light Integral" in s:
            return None
        if "Water Applied" in s:
            return None

        return None


    def _row_allows_mapping(row_label: str) -> bool:
        return _summary_row_to_canon(row_label) is not None


    def _summary_row_data_column(row_label: str) -> str:
        canon = _summary_row_to_canon(row_label)
        if canon is None:
            return LOCKED_DATA_COLUMN

        raw = canon_to_raw_saved.get(canon)
        if not raw:
            return UNMAPPED_DATA_COLUMN

        return raw_to_token.get(str(raw), UNMAPPED_DATA_COLUMN)

    def _target_pair_for_row(row_label: str):
        s = str(row_label)

        if "Air Temperature" in s:
            return target_temp_low, target_temp_high
        if "Leaf Temperature" in s:
            return target_temp_low, target_temp_high
        if "Relative Humidity" in s:
            return target_rh_low, target_rh_high
        if "Vapor Pressure Deficit" in s:
            return target_vpd_low, target_vpd_high
        if "Light Intensity" in s:
            return None, target_ppfd
        if "Daily Light Integral" in s:
            return target_dli_low, target_dli_high

        return None, None

    def _target_mode_for_row(row_label: str) -> str:
        s = str(row_label)

        if "Air Temperature" in s:
            return "both"
        if "Leaf Temperature" in s:
            return "both"
        if "Relative Humidity" in s:
            return "both"
        if "Vapor Pressure Deficit" in s:
            return "both"
        if "Daily Light Integral" in s:
            return "both"
        if "Light Intensity" in s:
            return "high_only"

        return "none"

    def _normalize_summary_editor_df(df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the editable summary table into a stable form so we can compare
        whether the user actually changed anything.
        """
        df2 = df_in.copy()

        # Treat blanks consistently
        for col in ["Data Column", "Metric"]:
            if col in df2.columns:
                df2[col] = df2[col].astype(str).fillna("")

        for col in ["Low Target", "High Target"]:
            if col in df2.columns:
                df2[col] = df2[col].apply(_safe_float)

        # Keep row order stable
        return df2.reset_index(drop=True)


    def _summary_editor_signature(df_in: pd.DataFrame) -> str:
        """
        Convert the editable summary table into a stable JSON string so that
        Streamlit reruns do not trigger unnecessary DB writes.
        """
        df2 = _normalize_summary_editor_df(df_in)

        return df2.to_json(
            orient="records",
            date_format="iso",
            default_handler=str,
        )

    #raw_options = [LOCKED_DATA_COLUMN] + [str(c) for c in raw_cols_for_dropdown]
    raw_options = [UNMAPPED_DATA_COLUMN] + list(token_to_raw.keys())

    INPUT_SEPARATOR_LABEL = "──── Input Metrics ────"
    CALC_SEPARATOR_LABEL = "──── Calculated Metrics ────"

    temp_symbol = "°F" if temp_unit == "F" else "°C"

    # Pick whichever VPD row exists
    vpd_candidates = [idx for idx in summary_display.index if "Vapor Pressure Deficit" in str(idx)]
    vpd_label = vpd_candidates[0] if vpd_candidates else "🍃 Leaf Vapor Pressure Deficit (kPa)"

    ordered_editor_rows = [
        INPUT_SEPARATOR_LABEL,
        f"🌡️ Air Temperature ({temp_symbol})",
        f"🍂 Leaf Temperature ({temp_symbol})",
        "💦 Relative Humidity (%)",
        "💡 Light Intensity (PPFD - µmol m⁻² s⁻¹)",
        f"🚿 {IRRIGATION_EVENTS_ROW_LABEL}",
        CALC_SEPARATOR_LABEL,
        vpd_label,
        "🌤️ Daily Light Integral (mol m⁻² d⁻¹)",
        "💧 Water Applied per Day (mL m⁻² d⁻¹)",
    ]

    editor_rows = []
    for row_label in ordered_editor_rows:
        if row_label in [INPUT_SEPARATOR_LABEL, CALC_SEPARATOR_LABEL]:
            editor_rows.append({
                "Metric": row_label,
                "Data Column": LOCKED_DATA_COLUMN,
                "Minimum": "",
                "Average": "",
                "Maximum": "",
                "Low Target": None,
                "High Target": None,
            })
        else:
            low_t, high_t = _target_pair_for_row(row_label)

            editor_rows.append({
                "Metric": row_label,
                "Data Column": _summary_row_data_column(row_label),
                "Minimum": summary_display.at[row_label, "Min"] if row_label in summary_display.index else "-",
                "Average": summary_display.at[row_label, "Average"] if row_label in summary_display.index else "-",
                "Maximum": summary_display.at[row_label, "Max"] if row_label in summary_display.index else "-",
                "Low Target": low_t,
                "High Target": high_t,
            })

    summary_editor_df = pd.DataFrame(editor_rows)


    # ---------- Build a style DataFrame for the editor ----------
    editor_style_df = pd.DataFrame(
        "",
        index=summary_editor_df.index,
        columns=summary_editor_df.columns,
    )

    # Reuse existing summary color logic
    metric_style_df = build_style_df(summary_display, summary_numeric)

    # Copy color styling from old summary table columns into new editor columns
    for i, metric_label in summary_editor_df["Metric"].items():
        if metric_label in metric_style_df.index:
            if "Min" in metric_style_df.columns and "Minimum" in editor_style_df.columns:
                editor_style_df.at[i, "Minimum"] = metric_style_df.at[metric_label, "Min"]
            if "Average" in metric_style_df.columns and "Average" in editor_style_df.columns:
                editor_style_df.at[i, "Average"] = metric_style_df.at[metric_label, "Average"]
            if "Max" in metric_style_df.columns and "Maximum" in editor_style_df.columns:
                editor_style_df.at[i, "Maximum"] = metric_style_df.at[metric_label, "Max"]

    # Keep non-output columns neutral
    for col in ["Metric", "Data Column", "Low Target", "High Target"]:
        if col in editor_style_df.columns:
            editor_style_df[col] = ""

    # ---------- Build styled editor ----------
    styled_summary_editor = (
        summary_editor_df.style
        .apply(lambda _: editor_style_df, axis=None)
        .set_properties(
            subset=["Minimum", "Average", "Maximum"],
            **{"text-align": "right"}
        )
        .format(
            {
                "Minimum": lambda x: "-" if pd.isna(x) or x == "" else str(x),
                "Average": lambda x: "-" if pd.isna(x) or x == "" else str(x),
                "Maximum": lambda x: "-" if pd.isna(x) or x == "" else str(x),
            }
        )
    )

    edited_summary = st.data_editor(
        styled_summary_editor,
        hide_index=True,
        width="stretch",
        #height="content",
        key=f"home_summary_editor_{int(rec['id'])}",
        column_config={
            "Metric": st.column_config.TextColumn("Metric", disabled=True, width="medium"),
            "Data Column": st.column_config.SelectboxColumn(
                "Data",
                options=raw_options,
                required=False,
                format_func=_format_data_column_option,
                help="Select which file column should map to this metric for this file. Rows showing '-' are calculated values and cannot be mapped.",
                width="medium",
            ),
            "Minimum": st.column_config.TextColumn("Minimum", disabled=True, width="small"),
            "Average": st.column_config.TextColumn("Average", disabled=True, width="small"),
            "Maximum": st.column_config.TextColumn("Maximum", disabled=True, width="small"),
            "Low Target": st.column_config.NumberColumn(
                "Low Target",
                required=False,
                format="%.1f",
                width="small",
            ),
            "High Target": st.column_config.NumberColumn(
                "High Target",
                required=False,
                format="%.1f",
                width="small",
            ),
        },
        disabled=["Metric", "Minimum", "Average", "Maximum"],
    )    

    for i, row in edited_summary.iterrows():
        metric = str(row["Metric"])

        # Data Column behavior
        if not _row_allows_mapping(metric):
            edited_summary.at[i, "Data Column"] = LOCKED_DATA_COLUMN
        else:
            current_val = edited_summary.at[i, "Data Column"]
            if current_val in (None, "", IGNORE_RAW):
                edited_summary.at[i, "Data Column"] = UNMAPPED_DATA_COLUMN

        # Target behavior
        target_mode = _target_mode_for_row(metric)

        if target_mode == "none":
            edited_summary.at[i, "Low Target"] = None
            edited_summary.at[i, "High Target"] = None
        elif target_mode == "high_only":
            edited_summary.at[i, "Low Target"] = None
    
    # ---------- AUTO-SAVE summary table ----------
    summary_sig_key = f"home_summary_editor_sig_{int(rec['id'])}"
    current_summary_sig = _summary_editor_signature(edited_summary)
    prev_summary_sig = st.session_state.get(summary_sig_key)

    if prev_summary_sig is None:
        # First render for this file: initialize only, do not write yet
        st.session_state[summary_sig_key] = current_summary_sig
    elif current_summary_sig != prev_summary_sig:
        # -------- 1) Save targets to settings --------
        def _row_by_contains(txt: str):
            m = edited_summary[edited_summary["Metric"].astype(str).str.contains(txt, regex=False)]
            return m.iloc[0] if not m.empty else None

        r_air  = _row_by_contains("Air Temperature")
        r_leaf = _row_by_contains("Leaf Temperature")
        r_rh   = _row_by_contains("Relative Humidity")
        r_vpd  = _row_by_contains("Vapor Pressure Deficit")
        r_par  = _row_by_contains("Light Intensity")
        r_dli  = _row_by_contains("Daily Light Integral")

        db.update_targets_from_summary_table(
            user["id"],
            target_low=_safe_float(r_air["Low Target"]) if r_air is not None else None,
            target_high=_safe_float(r_air["High Target"]) if r_air is not None else None,
            target_rh_low=_safe_float(r_rh["Low Target"]) if r_rh is not None else None,
            target_rh_high=_safe_float(r_rh["High Target"]) if r_rh is not None else None,
            target_ppfd=_safe_float(r_par["High Target"]) if r_par is not None else None,
            target_dli_low=_safe_float(r_dli["Low Target"]) if r_dli is not None else None,
            target_dli_high=_safe_float(r_dli["High Target"]) if r_dli is not None else None,
            target_vpd_low=_safe_float(r_vpd["Low Target"]) if r_vpd is not None else None,
            target_vpd_high=_safe_float(r_vpd["High Target"]) if r_vpd is not None else None,
        )

        # -------- 2) Save per-file canon_to_raw mapping --------
        new_canon_to_raw = dict(canon_to_raw_saved)

        for _, row in edited_summary.iterrows():
            canon = _summary_row_to_canon(row["Metric"])
            if canon is None:
                continue

            selected_token = row["Data Column"]
            if selected_token in (None, "", LOCKED_DATA_COLUMN, UNMAPPED_DATA_COLUMN):
                new_canon_to_raw[canon] = None
            else:
                new_canon_to_raw[canon] = token_to_raw.get(str(selected_token))

        db.upsert_file_column_map(
            user["id"],
            int(rec["id"]),
            raw_columns=[str(c) for c in raw_cols_for_dropdown],
            canon_to_raw=new_canon_to_raw,
            raw_preview_rows=None,
        )

        # -------- 3) Regenerate cleaned file bytes from raw upload --------
        raw_obj = db.get_raw_file_bytes(int(rec["id"]))
        if raw_obj is None:
            st.error(
                "This file does not have raw upload bytes saved, so the cleaned file cannot be regenerated. "
                "Re-upload the original file once to enable automatic summary-table updates."
            )
            st.stop()

        raw_filename = raw_obj["raw_filename"]
        ext = Path(raw_filename).suffix or ".csv"
        df_raw2, _, _, _ = load_table_from_bytes(raw_obj["bytes"], ext)

        raw_to_canon = _invert_canon_to_raw(new_canon_to_raw)
        df_clean2 = build_clean_dataframe(df_raw2, raw_to_canon)
        df_clean2 = apply_realistic_value_filters(
            df_clean2,
            orig_temp_unit=orig_temp_unit,
            show_warnings=True,
        )

        required_for_dashboard = ["Time", "AirTemp", "RH"]
        missing = [c for c in required_for_dashboard if c not in df_clean2.columns]
        if missing:
            st.warning(
                "Table changes were not applied because the dashboard file would be missing required columns: "
                + ", ".join(missing)
            )
            st.session_state[summary_sig_key] = current_summary_sig
            st.stop()

        cleaned_bytes2 = df_clean2.to_csv(index=False).encode("utf-8")
        db.update_file_content(int(rec["id"]), cleaned_bytes2)

        st.session_state[summary_sig_key] = current_summary_sig
        st.rerun()

else:
    st.info("No numeric columns found to summarize.")

# =========================
# Time Series (Dashboard graphs)
# =========================
#st.markdown("### Key Trends")
#st.subheader("Time series graphs")

use_time_axis = "Time" in df_display.columns and df_display["Time"].notna().any()
x_values = df_display["Time"] if use_time_axis else df_display.index

figs_for_pdf = []


# --- Summary-table page in PDF ---
def _pdf_ellipsis(text, max_chars: int) -> str:
    """
    Truncate long PDF table cell text so it stays inside the column.
    """
    s = "" if text is None else str(text)
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 3)] + "..."


def _pdf_fmt_value(val, decimals: int | None = None) -> str:
    """
    Formatter for PDF table cells.
    """
    if val is None:
        return "-"
    s = str(val).strip()
    if s == "" or s.lower() == "none" or s == "nan":
        return "-"
    if decimals is not None:
        try:
            return f"{float(val):.{decimals}f}"
        except Exception:
            return s
    return s


# --- Summary-table page in PDF ---
if summary is not None:
    fig_summary, ax_summary = plt.subplots(figsize=(12.2, 6.4))
    ax_summary.axis("off")
    fig_summary.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.04)

    # ---------- PDF header area ----------
    logo_path = Path("assets/EnDash_Logo_V3.png")

    summary_range_value = (
        f"{start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}"
        if start_time is not None and end_time is not None
        else "Unknown"
    )
    interval_value = interval_str if interval_td is not None else "Unknown"

    subtitle_color = "black"  #prv: dimgray
    subtitle_size = 11

    # Logo in top-left
    if logo_path.exists():
        try:
            logo_img = plt.imread(str(logo_path))
            logo_ax = fig_summary.add_axes([0.035, 0.950, 0.13, 0.10])  # left, bottom, width, height - prev: 0.015, 0.865, 0.11, 0.11
            logo_ax.imshow(logo_img)
            logo_ax.axis("off")
        except Exception:
            pass

    # Three gray lines to the right of the logo
    text_x = 0.18

    ax_summary.text(
        text_x, 1.060,
        f"Data Summary: {summary_range_value}",
        transform=ax_summary.transAxes,
        ha="left",
        va="bottom",
        fontsize=subtitle_size,
        color=subtitle_color,
    )

    ax_summary.text(
        text_x, 1.025,
        f"Data File: {filename}",
        transform=ax_summary.transAxes,
        ha="left",
        va="bottom",
        fontsize=subtitle_size,
        color=subtitle_color,
    )

    ax_summary.text(
        text_x, 0.990,
        f"Data Collection Interval: {interval_value}",
        transform=ax_summary.transAxes,
        ha="left",
        va="bottom",
        fontsize=subtitle_size,
        color=subtitle_color,
    )

    def _strip_pdf_emoji_only(s: str) -> str:
        """
        Remove leading emoji/icons used in the website table labels,
        but preserve scientific symbols like µ, ², ⁻, etc.
        """
        s = str(s).strip()

        emoji_prefixes = [
            "🌡️ ", "🍂 ", "💦 ", "🍃 ", "💡 ",
            "🌤️ ", "🚿 ", "💧 ", "🌿 "
        ]

        for prefix in emoji_prefixes:
            if s.startswith(prefix):
                return s[len(prefix):].strip()

        return s

    def _pdf_ellipsis(text, max_chars: int) -> str:
        s = "" if text is None else str(text)
        if len(s) <= max_chars:
            return s
        return s[: max(0, max_chars - 3)] + "..."

    # Build a PDF display frame that also includes the saved Data Column selections
    summary_pdf_display = summary_display.copy()
    summary_pdf_numeric = summary_numeric.copy()

    pdf_data_cols = []
    for row_label in summary_pdf_display.index:
        raw_val = _summary_row_data_column(row_label)
        pdf_data_cols.append(_format_data_column_option(raw_val))

    summary_pdf_display.insert(0, "Data Column", pdf_data_cols)
    summary_pdf_numeric.insert(0, "Data Column", np.nan)

    # Build color style table from the unstripped version
    style_df_pdf = build_style_df(summary_pdf_display, summary_pdf_numeric)

    # ---- Force PDF column order to match website table ----
    pdf_col_labels = [
        "Metric",
        "Data Column",
        "Min",
        "Average",
        "Max",
        "Low Target",
        "High Target",
    ]

    pdf_rows = []
    pdf_row_keys = []

    for row_label in summary_pdf_display.index:
        metric_text = _strip_pdf_emoji_only(str(row_label))

        # Keep metric fully shown as much as possible
        metric_text = _pdf_ellipsis(metric_text, 42)

        data_text = summary_pdf_display.at[row_label, "Data Column"]
        data_text = _pdf_ellipsis(data_text, 28)

        pdf_rows.append([
            metric_text,
            data_text,
            summary_pdf_display.at[row_label, "Min"],
            summary_pdf_display.at[row_label, "Average"],
            summary_pdf_display.at[row_label, "Max"],
            summary_pdf_display.at[row_label, "Low Target"],
            summary_pdf_display.at[row_label, "High Target"],
        ])
        pdf_row_keys.append(row_label)

    # Wider Metric column, moderate Data column, slimmer numeric columns
    col_widths = [0.31, 0.21, 0.09, 0.09, 0.09, 0.10, 0.11]

    tbl = ax_summary.table(
        cellText=pdf_rows,
        colLabels=pdf_col_labels,
        colWidths=col_widths,
        bbox=[0.01, 0.1, 0.98, 0.82],    #prev: 0.0, 0.02, 1.0, 0.90
        cellLoc="center",
        colLoc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.55)

    header_bg = "#F1F3F5"
    body_bg = "#FFFFFF"
    edge_color = "#D0D0D0"

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.8)

        if r == 0:
            cell.set_facecolor(header_bg)
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(10)
            cell.get_text().set_color("black")
            continue

        cell.set_facecolor(body_bg)

        row_label_original = pdf_row_keys[r - 1]

        # Metric column: left align, slightly less bold so long labels fit better
        if c == 0:
            cell.get_text().set_ha("left")
            cell.get_text().set_weight("semibold")
            cell.get_text().set_fontsize(8.5)

        # Data column: keep centered but smaller font to fit more text
        if c == 1:
            cell.get_text().set_fontsize(8.5)

        # Apply Min / Average / Max colors
        col_name_map = {
            1: "Data Column",
            2: "Min",
            3: "Average",
            4: "Max",
            5: "Low Target",
            6: "High Target",
        }

        if c in [2, 3, 4]:
            col_name = col_name_map[c]
            css = style_df_pdf.at[row_label_original, col_name]
            m = re.search(r"color:\s*([^;]+)", str(css))
            if m:
                cell.get_text().set_color(m.group(1).strip())

    figs_for_pdf.append(fig_summary)

# ---------- Special PPFD + DLI plot (dual Y-axes) ----------
numeric_cols_no_par = numeric_cols[:]
if "PAR" in numeric_cols:
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.set_ylabel("PPFD (µmol m⁻² s⁻¹)", color="tab:blue")
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
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
        zorder=2,
        label=f"Target PPFD Max ({target_ppfd:.0f})",
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
            target_dli_low,
            color="tab:orange",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
            label=f"Target DLI Low ({target_dli_low:.1f})",
        )
        ax2.axhline(
            target_dli_high,
            color="tab:orange",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
            label=f"Target DLI High ({target_dli_high:.1f})",
        )

    ax2.set_ylabel("DLI (mol m⁻² d⁻¹)", color="tab:orange")

    if use_time_axis:
        ax1.set_xlabel("Time")
        apply_time_axis_formatting(ax1, fig, x_values)
    else:
        ax1.set_xlabel("Index")
        ax1.xaxis.set_major_locator(plt.MaxNLocator(8))

    # Combined legend (reordered)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    handles = h1 + h2
    labels  = l1 + l2

    def _legend_sort_key(lbl: str) -> int:
        """
        Assign a priority so we can reorder legends robustly EVEN if some
        items are missing (e.g., DLI not computed -> no DLI legend entries).
        Desired order:
        1) PPFD
        2) Target PPFD
        3) DLI
        4) Target DLI
        5) anything else
        """
        s = (lbl or "").lower()

        # PPFD line
        if s.startswith("ppfd"):
            return 10

        # Target PPFD line
        if "target ppfd" in s:
            return 20

        # Target DLI lines
        if "target dli low" in s:
            return 35
        if "target dli high" in s:
            return 45

        # DLI bars (non-target)
        # (Make sure target DLI is caught above first)
        if s.startswith("dli"):
            return 30

        return 99

    # Sort by our key, but keep stable ordering among same-priority items
    sorted_items = sorted(zip(handles, labels), key=lambda hl: _legend_sort_key(hl[1]))
    handles_sorted, labels_sorted = zip(*sorted_items) if sorted_items else ([], [])

    ax1.legend(
        handles_sorted, labels_sorted,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.50),
        ncol=3,          # keeps a compact layout; works for 2 or 4 items
        frameon=True,
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.28)

    ax1.grid(True, linestyle=":", linewidth=0.5)

    st.pyplot(fig)
    plot_separator()
    figs_for_pdf.append(fig)

numeric_cols_no_par = [c for c in numeric_cols if c != "PAR"]
numeric_cols_no_par = [c for c in numeric_cols_no_par if c not in irrigation_cols]
LEAF_WETNESS_EXCLUDE = {
    "LeafWetness",
    "IrrigationEvents_LeafWetness",
}
numeric_cols_no_par = [c for c in numeric_cols_no_par if c not in LEAF_WETNESS_EXCLUDE]

# ---------- Generic plots for remaining numeric columns ----------
for col in numeric_cols_no_par:
    fig, ax = plt.subplots(figsize=(8, 3), dpi=200)
    y = df_display[col]

    # Temperature target bands
    if col in ["AirTemp", "LeafTemp"]:
        ax.axhline(
            target_temp_high,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Target high temperature ({target_temp_high:.0f})",
            zorder=1,
            antialiased=False,
        )
        ax.axhline(
            target_temp_low,
            color="blue",
            linestyle="--",
            linewidth=1.0,
            label=f"Target low temperature ({target_temp_low:.0f})",
            zorder=1,
            antialiased=False,
        )


    # Relative humidity target band
    if col == "RH":
        ax.axhline(
            target_rh_high,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Target high Relative Humidity ({target_rh_high:.0f}%)",
            zorder=1,
            antialiased=False,
        )
        ax.axhline(
            target_rh_low,
            color="blue",
            linestyle="--",
            linewidth=1.0,
            label=f"Target low Relative Humidity ({target_rh_low:.0f}%)",
            zorder=1,
            antialiased=False,
        )
        

    # VPD target band
    if col in ["VPDair", "VPDleaf"]:
        ax.axhline(
            target_vpd_high,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Target high VPD ({target_vpd_high:.2f} kPa)",
            zorder=1,
            antialiased=False,
        )
        ax.axhline(
            target_vpd_low,
            color="blue",
            linestyle="--",
            linewidth=1.0,
            label=f"Target low VPD ({target_vpd_low:.2f} kPa)",
            zorder=1,
            antialiased=False,
        )


    # Color the line based on target bands (black within, red above, blue below)
    has_band = col in ["AirTemp", "LeafTemp", "RH", "VPDair", "VPDleaf"]

    if has_band and y is not None and y.notna().any():
        if col in ["AirTemp", "LeafTemp"]:
            low, high = float(target_temp_low), float(target_temp_high)
        elif col == "RH":
            low, high = float(target_rh_low), float(target_rh_high)
        else:  # VPDair / VPDleaf
            low, high = float(target_vpd_low), float(target_vpd_high)

        #temporary fix for missing values around target lines
        ax.plot(x_values,y,color="black", linewidth=1.2,label=pretty_label(col, temp_unit),zorder=3)

        below = y < low
        above = y > high
        within = ~(below | above)

        #ax.plot(x_values, y.where(within), color="black", linewidth=1.2, label=pretty_label(col, temp_unit), zorder=3, solid_capstyle="butt", solid_joinstyle="miter")
        ax.plot(x_values, y.where(above), color="red", linewidth=1.2, zorder=4, solid_capstyle="butt", solid_joinstyle="miter")
        ax.plot(x_values, y.where(below), color="blue", linewidth=1.2, zorder=4, solid_capstyle="butt", solid_joinstyle="miter")
    else:
        ax.plot(x_values, y, label=pretty_label(col, temp_unit), zorder=3, solid_capstyle="butt", solid_joinstyle="miter")

    if use_time_axis:
        ax.set_xlabel("Time")
        apply_time_axis_formatting(ax, fig, x_values)
    else:
        ax.set_xlabel("Index")
        ax.xaxis.set_major_locator(plt.MaxNLocator(8))

    y_label = pretty_label(col, temp_unit)
    ax.set_ylabel(y_label)
    ax.set_title(y_label)

    legend_below(ax, fig, ncol=3, y=-0.5) #Set number of columns in time series graph legend
    #ax.grid(True, linestyle=":", linewidth=0.5)
    
    st.pyplot(fig)
    plot_separator()
    figs_for_pdf.append(fig)

# ----------------------------------------------------------
# Irrigation settings (moved from old Settings page)
# ----------------------------------------------------------
st.markdown("---")
st.subheader("Irrigation Settings")
st.caption(
    "These settings control how EnDash converts irrigation signal data into ON/OFF states "
    "and how it calculates water applied per day."
)

with st.form("homepage_irrigation_settings_form", clear_on_submit=False):
    col_it, col_gap = st.columns(2)

    with col_it:
        irrigation_trigger_input = st.number_input(
            "Irrigation Trigger (ON when value ≥ trigger)",
            value=float(irrigation_trigger),
            min_value=0.1,
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
                "multiple time steps."
            ),
        )

    water_applied_per_event_ml_m2_input = st.number_input(
        "Water Applied per Irrigation Event (mL m⁻²)",
        min_value=0.0,
        step=1.0,
        value=float(water_applied_per_event_ml_m2),
        format="%.0f",
        help="Used to calculate Water Applied per Day.",
    )

    save_irrig_settings = st.form_submit_button("💾 Save irrigation settings")

if save_irrig_settings:
    db.update_settings(
        user["id"],
        orig_temp_unit=orig_temp_unit,
        orig_light_unit=orig_light_unit,
        temp_unit=temp_unit,
        target_low=float(target_temp_low),
        target_high=float(target_temp_high),
        target_rh_low=float(target_rh_low),
        target_rh_high=float(target_rh_high),
        target_ppfd=float(target_ppfd),
        target_dli_low=float(target_dli_low),
        target_dli_high=float(target_dli_high),
        target_vpd_low=float(target_vpd_low),
        target_vpd_high=float(target_vpd_high),
        irrigation_trigger=float(irrigation_trigger_input),
        irrigation_min_interval_min=float(irrigation_min_interval_input),
        leaf_wetness_unit=leaf_wetness_unit,
        irrigation_sensitivity_pct=float(irrigation_sensitivity_pct),
        leaf_wetness_min_interval_min=float(leaf_wetness_min_interval_min),
        water_applied_per_event_ml_m2=float(water_applied_per_event_ml_m2_input),
    )
    st.success("Irrigation settings saved.")
    st.rerun()


# ----------------------------------------------------------
# Irrigation plots (PER ZONE)
# ----------------------------------------------------------
if use_time_axis and events_by_zone is not None and not events_by_zone.empty:
    cols = list(events_by_zone.columns)
    n = len(cols)

    # --- Plot 1: Events per Day (grouped bars by zone) ---
    fig_ir, ax_ir = plt.subplots(figsize=(8, 3))
    ax_ir.set_title("Irrigation Events per Day (Full Days Only)")
    ax_ir.set_ylabel("Events per day")
    ax_ir.set_xlabel("Date")

    base_x = events_by_zone.index + pd.Timedelta(hours=12)

    # width in "days" units for datetime bars
    width = 0.8 / max(n, 1)

    for i, col in enumerate(cols):
        # center the grouped bars around the day midpoint
        offset_days = (i - (n - 1) / 2) * width
        x = base_x + pd.to_timedelta(offset_days, unit="D")
        ax_ir.bar(x, events_by_zone[col].values, width=width * 0.95, align="center", label=col)

    apply_time_axis_formatting(ax_ir, fig_ir, events_by_zone.index)
    #legend_below(ax_ir, fig_ir, ncol=min(3, n), y=-0.5)

    st.pyplot(fig_ir)
    plot_separator()
    figs_for_pdf.append(fig_ir)

    # --- Plot 2+: ON/OFF (Binary) per zone for a selected full day (auto-updates) ---
    IRR_DAY_KEY = "dash_irrigation_day_to_graph"

    full_days_ir = irrig_stats.get("full_days", []) or []
    irr_day_options = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in full_days_ir] if full_days_ir else []

    if irr_day_options:
        # init default (latest full day) if missing/invalid (important when switching files)
        if (IRR_DAY_KEY not in st.session_state) or (st.session_state[IRR_DAY_KEY] not in irr_day_options):
            st.session_state[IRR_DAY_KEY] = irr_day_options[-1]

        st.selectbox(
            "Day to Graph (Irrigation 24hr)",
            options=irr_day_options,
            key=IRR_DAY_KEY,
            help="Select a full day to display the 24hr irrigation ON/OFF graphs.",
        )

        day_to_plot = pd.to_datetime(st.session_state[IRR_DAY_KEY])
    else:
        st.info("No full days detected for irrigation 24hr plots.")
        day_to_plot = None

    if day_to_plot is not None:
        # Build binary series per zone for the selected day
        day_mask = df_display["Time"].dt.normalize() == day_to_plot.normalize()

        for col in cols:
            sig = pd.to_numeric(df_display.loc[day_mask, col], errors="coerce").fillna(0.0)
            on = (sig >= float(irrigation_trigger)).astype(int)

            day_df = pd.DataFrame(
                {"Time": df_display.loc[day_mask, "Time"].values, "IrrigationOn": on.values}
            )

            if day_df.empty:
                continue

            fig_day, ax_day = plt.subplots(figsize=(8, 3))
            ax_day.set_title(f"{col} ON/OFF Over 24 Hours — {day_to_plot.date()}")
            ax_day.set_ylabel("Irrigation (0=Off, 1=On)")
            ax_day.set_xlabel("Time of day")

            ax_day.step(day_df["Time"], day_df["IrrigationOn"], where="post", linewidth=1.5)
            ax_day.set_ylim(-0.1, 1.1)

            apply_time_axis_formatting(ax_day, fig_day, day_df["Time"])

            st.pyplot(fig_day)
            plot_separator()
            figs_for_pdf.append(fig_day)



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
        label="⬇️ Download Report",
        data=pdf_buffer,
        file_name=f"endash_{Path(filename).stem}.pdf",
        mime="application/pdf",
        width="stretch",
    )

    # Close figures after rendering + PDF generation
    for fig in figs_for_pdf:
        plt.close(fig)
else:
    download_slot.button("⬇️ Download Report", disabled=True, width="stretch")

st.markdown("---")
st.caption("Courtesy of the Fisher Lab - IFAS, University of Florida")
