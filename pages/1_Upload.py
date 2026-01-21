# pages/1_Upload.py

import io
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from lib import auth, db

import csv
from io import StringIO

import hashlib

# Adding numbers to column names
def make_indexed_labels(raw_cols: List[str]) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Create dropdown/display labels like: '[03] Air Temperature'
    Returns:
      labels: list of labeled strings (same order as raw_cols)
      raw_to_label: {raw_name -> label}
      label_to_raw: {label -> raw_name}
    """
    labels = [f"[{i:02d}] {c}" for i, c in enumerate(raw_cols)]
    raw_to_label = {raw: f"[{i:02d}] {raw}" for i, raw in enumerate(raw_cols)}
    label_to_raw = {v: k for k, v in raw_to_label.items()}
    return labels, raw_to_label, label_to_raw


def with_indexed_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with headers renamed to '[##] name'."""
    cols = [str(c) for c in df.columns]
    new_cols = {c: f"[{i:02d}] {c}" for i, c in enumerate(cols)}
    return df.rename(columns=new_cols)

def missing_required_cols(df_clean: pd.DataFrame) -> List[str]:
    """
    Require:
      1) A Time column
      2) At least one additional (non-Time) data column
    Return a list of "missing requirements" for display.
    """
    missing = []

    has_time = "Time" in df_clean.columns
    if not has_time:
        missing.append("Time")

    # Count non-Time columns with at least one non-null value
    data_cols = [c for c in df_clean.columns if c != "Time"]
    has_any_data_col = any(df_clean[c].notna().any() for c in data_cols)

    if not has_any_data_col:
        missing.append("At least one data column")

    return missing

def show_incomplete_mapping_warning(missing: List[str]) -> None:
    st.warning(
        "Mapping updates the preview immediately, but the dashboard file is **not updated yet**.\n\n"
        f"Missing requirement(s): **{', '.join(missing)}**\n\n"
        "Map a Time column and at least one data column to enable dashboard updates.",
        icon="⚠️",
    )

def mapping_signature(canon_list: List[str], canon_to_raw: Dict[str, Optional[str]]) -> str:
    parts = [f"{c}={canon_to_raw.get(c) or ''}" for c in canon_list]
    return "|".join(parts)

def preview_table_with_mapping_row(
    df_raw_preview: pd.DataFrame,
    raw_cols: List[str],
    canon_to_raw: Dict[str, Optional[str]],
    none_label: str = "(None)",
) -> pd.DataFrame:
    """
    Build a single preview table (like raw preview) but with a top row showing mapped canonical names.
    The row values are the canonical label for each raw column, or '(None)' if unmapped.
    """
    # Invert canon_to_raw -> raw_to_canon (one-to-one is expected)
    raw_to_canon = {str(raw): str(canon) for canon, raw in (canon_to_raw or {}).items() if raw}

    # Mapping row aligned to raw_cols
    mapping_row = {c: raw_to_canon.get(str(c), none_label) for c in raw_cols}

    df_head = df_raw_preview.head(10).copy()

    # Prepend mapping row
    df_show = pd.concat([pd.DataFrame([mapping_row]), df_head], ignore_index=True)

    # Make index label nicer (Mapped row + data rows)
    df_show.index = ["Mapped as"] + [str(i) for i in range(len(df_head))]


    df_show = make_arrow_safe_for_preview(df_show)

    # Add indexed headers for readability
    return with_indexed_headers(df_show)

def make_arrow_safe_for_preview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit/Arrow-safe dataframe for display:
    - Decodes bytes -> strings
    - Converts all cells -> strings ('' for NA)
    Uses DataFrame.map (pandas >=2.1/2.2) with a safe fallback.
    """
    def _cell(x):
        if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
            return ""
        if isinstance(x, (bytes, bytearray, memoryview)):
            try:
                return bytes(x).decode("utf-8", errors="replace")
            except Exception:
                return str(x)
        return str(x)

    # pandas new API: DataFrame.map
    if hasattr(df, "map"):
        return df.map(_cell)

    # fallback for older pandas: apply column-wise map
    return df.apply(lambda s: s.map(_cell))


# ------------- Canonical mapping helpers -------------

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
    "LeafWetness": ["leaf wetness", "leafwetness", "leaf_wetness", "lw","leaf wetness %", "leaf wetness (%)", "leaf wetness (v)"],
    "Date": ["date", "day", "log_date", "recorded_date"],
    "TimeOfDay": ["time_of_day", "clock", "clock_time", "timeofday", "time only", "time_only"],
}

MAX_IRRIGATION_ZONES = 5

# What we *output* in the cleaned CSV
CANON_OUTPUT_BASE = ["Time", "AirTemp", "LeafTemp", "RH", "PAR", "LeafWetness"]
IRR_CANONS = [f"Irrigation{i}" for i in range(1, MAX_IRRIGATION_ZONES + 1)]
CANON_OUTPUT_ORDER = CANON_OUTPUT_BASE + IRR_CANONS

# What we show in the mapping UI (includes optional Date/TimeOfDay)
CANON_UI_BASE = ["Time", "Date", "TimeOfDay", "AirTemp", "LeafTemp", "RH", "PAR", "LeafWetness"]

def build_alias_table() -> Dict[str, set]:
    table = {}
    for canon, aliases in ALIASES.items():
        table[canon] = {normalize(canon), *[normalize(a) for a in aliases]}
    return table


def map_columns(raw_cols: List[str], alias_table: Dict[str, set]) -> Tuple[Dict[str, str], List[str], List[str]]:
    norm_to_raw = {normalize(c): c for c in raw_cols}
    mapping, used = {}, set()

    for canon, norms in alias_table.items():
        for norm, raw in norm_to_raw.items():
            if norm in norms and raw not in used:
                mapping[raw] = canon
                used.add(raw)
                break

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

def _score_header_candidate(values: List[object], alias_table: Dict[str, set]) -> float:
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

    # small boost if row contains at least one "date" and one "time"-ish token
    norms = [normalize(s) for s in cleaned]
    has_date = any(n in ("date", "day") or "date" in n for n in norms)
    has_time = any(n == "time" or "time" in n for n in norms)

    score = 0.0
    score += len(cleaned) * 1.0
    score += uniq_ratio * 2.0
    score += hits * 5.0
    score += 2.0 if (has_date and has_time) else 0.0

    return score


def detect_header_row_from_preview(df_preview: pd.DataFrame, alias_table: Dict[str, set]) -> int:
    """
    Given a preview df read with header=None, return best header row index.
    Defaults to 0 if nothing clearly wins.
    """
    best_row = 0
    best_score = -1.0

    # Only scan first N rows of preview
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

def load_table_from_bytes(file_bytes: bytes, ext: str) -> Tuple[pd.DataFrame, str, Optional[str], int]:
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
            preview, delim = preview_csv_to_dataframe(file_bytes, enc, n_lines=10)
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


def build_clean_dataframe(df_raw: pd.DataFrame, raw_to_canon: Dict[str, str]) -> pd.DataFrame:
    canon_to_raw: Dict[str, str] = {}
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
    base = user.get("username") or user.get("email", "").split("@")[0] or f"user{user['id']}"
    slug = re.sub(r"[^a-zA-Z0-9]+", "", base).lower()
    return slug or f"user{user['id']}"


def canon_select_ui(
    *,
    raw_cols: List[str],
    default_canon_to_raw: Dict[str, Optional[str]],
    form_key: str,
    canon_list: List[str],
) -> Tuple[Dict[str, str], Dict[str, Optional[str]], List[str]]:
    """
    Builds the dropdown UI for selecting canonical columns.
    canon_list controls which dropdowns appear (lets us show Irrigation1..N).
    """
    labeled_cols, raw_to_label, label_to_raw = make_indexed_labels(raw_cols)

    options_all = ["(None)"] + labeled_cols
    canon_to_raw: Dict[str, Optional[str]] = {}

    for canon in canon_list:
        default_raw = default_canon_to_raw.get(canon)
        default_label = raw_to_label.get(default_raw) if default_raw else None
        default_index = options_all.index(default_label) if default_label in options_all else 0

        sel_label = st.selectbox(
            f"{canon} column",
            options_all,
            index=default_index,
            key=f"{form_key}_{canon}",
        )

        canon_to_raw[canon] = None if sel_label == "(None)" else label_to_raw[sel_label]

    # Build raw_to_canon (the mapping that cleaning uses)
    raw_to_canon: Dict[str, str] = {}
    used = set()
    duplicates = []

    for canon, raw in canon_to_raw.items():
        if not raw:
            continue
        if raw in used:
            duplicates.append(raw)
        else:
            used.add(raw)
            raw_to_canon[raw] = canon

    return raw_to_canon, canon_to_raw, sorted(set(duplicates))

# ---- New UI: "mapping header row" editor (one-row data_editor) ----

IGNORE_CANON = "(Ignore)"

def _invert_canon_to_raw_to_raw_to_canon(canon_to_raw: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Convert {canon: raw} -> {raw: canon} (dropping Nones)."""
    out = {}
    for canon, raw in (canon_to_raw or {}).items():
        if raw:
            out[str(raw)] = str(canon)
    return out


def mapping_header_editor(
    *,
    raw_cols: List[str],
    default_raw_to_canon: Dict[str, str],
    canon_options: List[str],
    editor_key: str,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Shows a 1-row grid where each RAW column is a cell, and the cell value is the selected CANON name.
    Returns:
      raw_to_canon: {raw_col: canon} for non-ignored selections
      dup_canons: list of canonical names that were selected more than once
    """
    # Use indexed labels so columns are visually distinct and stable
    labels, raw_to_label, label_to_raw = make_indexed_labels(raw_cols)

    row_dict = {}
    for raw in raw_cols:
        lab = raw_to_label[raw]
        row_dict[lab] = default_raw_to_canon.get(raw, IGNORE_CANON)

    df_row = pd.DataFrame([row_dict], index=["Mapped as"])

    col_cfg = {
        lab: st.column_config.SelectboxColumn(
            options=canon_options,
            required=False,
            help="Select the canonical meaning for this raw column."
        )
        for lab in labels
    }

    edited = st.data_editor(
        df_row,
        column_config=col_cfg,
        hide_index=False,
        width='stretch',
        key=editor_key,
    )

    # Build raw_to_canon from edited row
    raw_to_canon: Dict[str, str] = {}
    selected_canons = []
    for lab in labels:
        raw = label_to_raw[lab]
        val = str(edited.iloc[0][lab]) if edited.iloc[0][lab] is not None else IGNORE_CANON
        if val and val != IGNORE_CANON:
            raw_to_canon[raw] = val
            selected_canons.append(val)

    # Detect duplicate canon selections (not allowed)
    dup_canons = sorted({c for c in selected_canons if selected_canons.count(c) > 1})
    return raw_to_canon, dup_canons


def save_mapping_and_regenerate_cleaned_file(
    *,
    user_id: int,
    file_id: int,
    raw_cols: List[str],
    raw_to_canon: Dict[str, str],
    raw_preview_rows: Optional[List[Dict[str, object]]] = None,
) -> Tuple[bool, str]:
    """
    Persists mapping + regenerates cleaned bytes in Neon (if raw bytes exist).
    Returns (ok, message).
    """
    # Convert raw_to_canon -> canon_to_raw for persistence
    canon_to_raw = {canon: raw for raw, canon in raw_to_canon.items()}

    raw_obj = db.get_raw_file_bytes(file_id)
    if not raw_obj:
        # Still save mapping metadata + template, but can't regenerate cleaned file
        db.upsert_file_column_map(
            user_id,
            file_id,
            raw_columns=raw_cols,
            canon_to_raw=canon_to_raw,
            raw_preview_rows=raw_preview_rows,
        )
        db.set_last_upload_context(user_id, file_id=file_id, raw_columns=raw_cols, canon_to_raw=canon_to_raw)
        return False, (
            "Mapping saved, but this file was uploaded before raw-file storage was enabled, "
            "so the cleaned file cannot be regenerated automatically. Re-upload the original file once "
            "to enable live regeneration on future edits."
        )

    raw_filename = raw_obj["raw_filename"]
    ext = Path(raw_filename).suffix or ".csv"
    df_raw2, _, _, _ = load_table_from_bytes(raw_obj["bytes"], ext)

    df_clean2 = build_clean_dataframe(df_raw2, raw_to_canon)

    required_for_dashboard = ["Time", "AirTemp", "RH"]
    missing = [c for c in required_for_dashboard if c not in df_clean2.columns]
    if missing:
        return False, (
            "Cannot apply mapping because the regenerated cleaned file would be missing "
            "required dashboard columns: " + ", ".join(missing)
        )

    # Save mapping metadata (+ refreshed preview), overwrite cleaned bytes, persist template
    preview_rows = (
        df_raw2.head(10)
        .where(pd.notnull(df_raw2.head(10)), None)
        .to_dict(orient="records")
    )

    db.upsert_file_column_map(
        user_id,
        file_id,
        raw_columns=[str(c) for c in df_raw2.columns],
        canon_to_raw=canon_to_raw,
        raw_preview_rows=preview_rows,
    )

    cleaned_bytes2 = df_clean2.to_csv(index=False).encode("utf-8")
    db.update_file_content(file_id, cleaned_bytes2)

    db.set_last_upload_context(user_id, file_id=file_id, raw_columns=[str(c) for c in df_raw2.columns], canon_to_raw=canon_to_raw)
    return True, "Mapping saved and cleaned file regenerated."

# ------------- Streamlit page -------------

st.set_page_config(page_title="Upload", page_icon="📂", layout="wide")
auth.require_login()
user = auth.current_user()
auth.render_sidebar()

db.init_db()

st.title("📂 Upload data file")

uploaded = st.file_uploader("Upload a new data file", type=["csv", "xlsx", "xls", "xlsm"])
new_upload_active = uploaded is not None

# ---- Previously uploaded file selector (persistent) ----
#st.caption("Select a previously uploaded cleaned file to review/download it and manage its saved column mapping.")
files = db.list_user_files(user["id"])

last_ctx = db.get_last_upload_context(user["id"])
last_file_id = last_ctx.get("file_id")

options = {f"{rec['filename']} ({rec['uploaded_at']})": rec for rec in files}

selected_label = None
selected_file_rec = None
selected_file_id = None

if options:
    labels = list(options.keys())
    default_index = 0
    if last_file_id is not None:
        for i, lab in enumerate(labels):
            if int(options[lab]["id"]) == int(last_file_id):
                default_index = i
                break

    selected_label = st.selectbox(
        "Current File Selection",
        labels,
        index=default_index,
        key="uploaded_file_select",
    )
    selected_file_rec = options.get(selected_label)
    selected_file_id = int(selected_file_rec["id"])
else:
    st.info("No cleaned files uploaded yet. Upload a file above to get started.")

st.divider()

# ---- Unified: Active dataset (uploaded OR selected existing) ----

active_mode = None          # "upload" or "existing"
active_key = None           # used for session keys
active_file_id = None       # Neon file_id if exists
df_raw_full = None          # full raw df for cleaning
df_raw_preview = None       # raw df for preview
raw_cols = None
saved_canon_to_raw = {}
raw_bytes_available = False
raw_bytes = None
raw_filename = None

if uploaded is not None:
    # ---------- New upload becomes active dataset ----------
    active_mode = "upload"
    raw_bytes = uploaded.getvalue()
    raw_filename = uploaded.name
    ext = Path(raw_filename).suffix or ".csv"

    df_raw_full, file_type, encoding_used, header_row = load_table_from_bytes(raw_bytes, ext)
    df_raw_preview = df_raw_full
    raw_cols = [str(c) for c in df_raw_full.columns]
    raw_bytes_available = True

    # stable key per upload
    fp = hashlib.md5(raw_bytes).hexdigest()[:12]
    active_key = f"upload_{fp}"

    # reset per-upload state when a different file is uploaded
    if st.session_state.get("_active_upload_key") != active_key:
        st.session_state["_active_upload_key"] = active_key
        st.session_state.pop(f"{active_key}_file_id", None)
        st.session_state.pop(f"{active_key}_sig", None)

    active_file_id = st.session_state.get(f"{active_key}_file_id")

else:
    # ---------- Existing selection becomes active dataset ----------
    if selected_file_id is None:
        st.info("Upload a file above to begin, or select a previously uploaded file.", icon="⬆️")
        st.stop()

    active_mode = "existing"
    active_file_id = selected_file_id
    active_key = f"file_{selected_file_id}"

    map_rec = db.get_file_column_map(selected_file_id) or {}
    raw_cols = [str(c) for c in (map_rec.get("raw_columns") or [])]
    saved_canon_to_raw = (map_rec.get("canon_to_raw") or {})

    # Try to get raw bytes (needed for regeneration)
    raw_obj = db.get_raw_file_bytes(selected_file_id)
    if raw_obj:
        raw_bytes_available = True
        raw_filename = raw_obj["raw_filename"]
        ext = Path(raw_filename).suffix or ".csv"
        df_raw_full, _, _, _ = load_table_from_bytes(raw_obj["bytes"], ext)
        df_raw_preview = df_raw_full
        if not raw_cols:
            raw_cols = [str(c) for c in df_raw_full.columns]
    else:
        raw_bytes_available = False
        # fallback preview from stored preview rows
        if map_rec.get("raw_preview"):
            df_raw_preview = pd.DataFrame(map_rec["raw_preview"])
            if not raw_cols:
                raw_cols = [str(c) for c in df_raw_preview.columns]

    if df_raw_preview is None or not raw_cols:
        st.warning("No raw preview available for this dataset. Re-upload the original file to enable mapping edits.")
        st.stop()


# ---------- UI: single preview table at top ----------
preview_ph = st.empty()

# ---------- Defaults for mapping dropdowns ----------
alias_table = build_alias_table()
auto_raw_to_canon, _, _ = map_columns(raw_cols, alias_table)
auto_canon_to_raw = {}
for raw, canon in auto_raw_to_canon.items():
    auto_canon_to_raw.setdefault(canon, raw)

default_canon_to_raw = dict(auto_canon_to_raw)

# overlay "template" from last context (if available)
template = (last_ctx.get("canon_to_raw") or {})
for canon, raw in template.items():
    if raw and raw in raw_cols:
        default_canon_to_raw[canon] = raw

# overlay saved mapping for existing files
if active_mode == "existing":
    for canon, raw in (saved_canon_to_raw or {}).items():
        if raw and raw in raw_cols:
            default_canon_to_raw[canon] = raw

# irrigation zones count
irrig_state_key = f"{active_key}_irrig_count"
if irrig_state_key not in st.session_state:
    inferred = 1
    for i in range(1, MAX_IRRIGATION_ZONES + 1):
        if (default_canon_to_raw or {}).get(f"Irrigation{i}"):
            inferred = i
    st.session_state[irrig_state_key] = inferred


canon_list = CANON_UI_BASE + [f"Irrigation{i}" for i in range(1, st.session_state[irrig_state_key] + 1)]

st.markdown("#### Select mapped column names")
st.caption("Your selections update the preview immediately. Dashboard updates occur automatically once required columns are mapped.")

raw_to_canon, canon_to_raw, duplicates = canon_select_ui(
    raw_cols=raw_cols,
    default_canon_to_raw=default_canon_to_raw,
    form_key=f"map_{active_key}",
    canon_list=canon_list,
)

# update the single preview table (with top mapping row)
preview_ph.dataframe(
    preview_table_with_mapping_row(df_raw_preview, raw_cols, canon_to_raw),
    width='stretch',
)

if duplicates:
    st.error("Duplicate selections: " + ", ".join(duplicates))
    st.stop()

# ---------- AUTO-SAVE + AUTO-REGENERATE ----------
sig_key = f"{active_key}_sig"
current_sig = mapping_signature(canon_list, canon_to_raw)
prev_sig = st.session_state.get(sig_key)

if prev_sig is None:
    # first render: initialize only
    st.session_state[sig_key] = current_sig
else:
    if current_sig != prev_sig:
        # Always build clean df from FULL raw data if we can
        if df_raw_full is None:
            # Can't regenerate without full df/raw bytes, but for existing we can still persist mapping
            if active_mode == "existing":
                db.upsert_file_column_map(
                    user["id"],
                    active_file_id,
                    raw_columns=raw_cols,
                    canon_to_raw=canon_to_raw,
                    raw_preview_rows=df_raw_preview.head(10).where(pd.notnull(df_raw_preview.head(10)), None).to_dict(orient="records"),
                )
                db.set_last_upload_context(user["id"], file_id=active_file_id, raw_columns=raw_cols, canon_to_raw=canon_to_raw)
                st.toast("Mapping saved. Re-upload this dataset once to enable dashboard regeneration.", icon="⚠️")
            st.session_state[sig_key] = current_sig
            st.stop()

        df_clean = build_clean_dataframe(df_raw_full, raw_to_canon)
        missing = missing_required_cols(df_clean)

        # If we have a file_id already, save mapping progress immediately
        if active_file_id is not None:
            db.upsert_file_column_map(
                user["id"],
                active_file_id,
                raw_columns=raw_cols,
                canon_to_raw=canon_to_raw,
                raw_preview_rows=df_raw_full.head(10).where(pd.notnull(df_raw_full.head(10)), None).to_dict(orient="records"),
            )
            db.set_last_upload_context(user["id"], file_id=active_file_id, raw_columns=raw_cols, canon_to_raw=canon_to_raw)

        if missing:
            # Warning banner (no dashboard update yet)
            show_incomplete_mapping_warning(missing)
            st.session_state[sig_key] = current_sig
            st.stop()

        # Mapping is valid — create/update dashboard file automatically
        cleaned_bytes = df_clean.to_csv(index=False).encode("utf-8")

        if active_mode == "upload" and active_file_id is None:
            # Create file record now that mapping is valid
            uname = username_slug(user)

            orig_stem = Path(raw_filename).stem
            orig_stem = re.sub(r"\s+", "_", orig_stem)
            orig_stem = re.sub(r"[^A-Za-z0-9_-]+", "", orig_stem)
            orig_stem = re.sub(r"_+", "_", orig_stem).strip("_") or "file"

            stored_filename = f"{orig_stem}_{uname}.csv"

            active_file_id = db.add_file_record(
                user["id"],
                stored_filename,
                cleaned_bytes,
                raw_filename=raw_filename,
                raw_bytes=raw_bytes,
            )
            st.session_state[f"{active_key}_file_id"] = active_file_id

            # persist mapping metadata + template
            db.upsert_file_column_map(
                user["id"],
                active_file_id,
                raw_columns=raw_cols,
                canon_to_raw=canon_to_raw,
                raw_preview_rows=df_raw_full.head(10).where(pd.notnull(df_raw_full.head(10)), None).to_dict(orient="records"),
            )
            db.set_last_upload_context(user["id"], file_id=active_file_id, raw_columns=raw_cols, canon_to_raw=canon_to_raw)

            st.toast("Dashboard file created and updated.", icon="✅")
            st.session_state[sig_key] = current_sig
            st.rerun()

        else:
            # Existing file OR already-created upload file: update dashboard bytes
            if active_mode == "existing" and not raw_bytes_available:
                st.warning("Cannot update dashboard file because raw bytes are not available. Re-upload the original file once.", icon="⚠️")
            else:
                db.update_file_content(active_file_id, cleaned_bytes)
                st.toast("Dashboard file updated.", icon="✅")

            st.session_state[sig_key] = current_sig

b1, b2 = st.columns([1, 1])
with b1:
    if st.button("Add irrigation zone", key=f"{active_key}_add_irrig"):
        st.session_state[irrig_state_key] = min(MAX_IRRIGATION_ZONES, st.session_state[irrig_state_key] + 1)
        st.rerun()
with b2:
    if st.button("Remove irrigation zone", key=f"{active_key}_rm_irrig"):
        st.session_state[irrig_state_key] = max(1, st.session_state[irrig_state_key] - 1)
        st.rerun()