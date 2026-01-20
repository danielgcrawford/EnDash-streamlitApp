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

# ---- Section A: existing file review/edit ----
if (not new_upload_active) and (selected_file_id is not None):
    st.subheader("Saved mapping + file preview")

    map_rec = db.get_file_column_map(selected_file_id)

    if map_rec and map_rec.get("canon_to_raw") is not None:
        db.set_last_upload_context(
            user["id"],
            file_id=selected_file_id,
            raw_columns=map_rec.get("raw_columns") or [],
            canon_to_raw=map_rec.get("canon_to_raw") or {},
        )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Cleaned file preview (stored in Neon)")
        file_obj = db.get_file_bytes(selected_file_id)
        if not file_obj:
            st.error("Could not load file bytes from Neon.")
        else:
            df_clean = pd.read_csv(io.BytesIO(file_obj["bytes"]))
            st.dataframe(df_clean.head(10), use_container_width=True)
            st.caption(f"Detected columns: {', '.join(df_clean.columns)}")

            st.download_button(
                "⬇️ Download cleaned CSV",
                data=file_obj["bytes"],
                file_name=file_obj["filename"],
                mime="text/csv",
            )

    with col2:
        st.markdown("#### Saved column mapping (editable)")
        if not map_rec:
            st.info("No mapping metadata found for this file (older upload).")
        else:
            raw_cols = [str(c) for c in (map_rec.get("raw_columns") or [])]
            saved_canon_to_raw = map_rec.get("canon_to_raw") or {}

            if not raw_cols:
                st.warning("No raw column list was saved for this file; selections cannot be edited.")
            else:
                # ---------- Irrigation UI state (Saved mapping editor) ----------
                if "savedmap_irrig_count" not in st.session_state:
                    # infer from saved mapping keys, default to 1
                    inferred = 1
                    for i in range(1, MAX_IRRIGATION_ZONES + 1):
                        if (saved_canon_to_raw or {}).get(f"Irrigation{i}"):
                            inferred = i
                    st.session_state.savedmap_irrig_count = inferred

                b1, b2 = st.columns([1, 1])
                with b1:
                    if st.button("Add irrigation zone", key="savedmap_add_irrig"):
                        st.session_state.savedmap_irrig_count = min(
                            MAX_IRRIGATION_ZONES, st.session_state.savedmap_irrig_count + 1
                        )
                with b2:
                    if st.button("Remove irrigation zone", key="savedmap_rm_irrig"):
                        st.session_state.savedmap_irrig_count = max(1, st.session_state.savedmap_irrig_count - 1)

                canon_list_saved = CANON_OUTPUT_BASE + [f"Irrigation{i}" for i in range(1, st.session_state.savedmap_irrig_count + 1)]

                with st.form("edit_saved_mapping_form"):
                    st.caption(
                        "These selections are saved with this file and also used as your default mapping template for Quick Upload."
                    )
                    _, canon_to_raw, duplicates = canon_select_ui(
                        raw_cols=raw_cols,
                        default_canon_to_raw=saved_canon_to_raw,
                        form_key="savedmap",
                        canon_list=canon_list_saved,
                    )
                    save_map = st.form_submit_button("💾 Save mapping selections")

                if save_map:
                    if duplicates:
                        st.error(f"Duplicate selections: {', '.join(duplicates)}")
                    else:
                        # 1) Try to regenerate cleaned file from stored RAW bytes
                        raw_obj = db.get_raw_file_bytes(selected_file_id)
                        if not raw_obj:
                            # Older files may not have raw bytes stored yet
                            db.upsert_file_column_map(
                                user["id"],
                                selected_file_id,
                                raw_columns=raw_cols,
                                canon_to_raw=canon_to_raw,
                                raw_preview_rows=map_rec.get("raw_preview"),
                            )
                            db.set_last_upload_context(
                                user["id"],
                                file_id=selected_file_id,
                                raw_columns=raw_cols,
                                canon_to_raw=canon_to_raw,
                            )
                            st.warning(
                                "Saved mapping selections, but this file was uploaded before raw-file storage was enabled, "
                                "so the cleaned file cannot be regenerated automatically. Re-upload the original file once "
                                "to enable live regeneration on future edits."
                            )
                        else:
                            # Convert canon_to_raw -> raw_to_canon for cleaning
                            raw_to_canon_new = {raw: canon for canon, raw in canon_to_raw.items() if raw}

                            # Re-read raw file exactly like the original upload flow
                            raw_filename = raw_obj["raw_filename"]
                            ext = Path(raw_filename).suffix or ".csv"
                            df_raw2, _, _, _ = load_table_from_bytes(raw_obj["bytes"], ext)

                            df_clean2 = build_clean_dataframe(df_raw2, raw_to_canon_new)

                            required_for_dashboard = ["Time", "AirTemp", "RH"]
                            missing = [c for c in required_for_dashboard if c not in df_clean2.columns]
                            if missing:
                                st.error(
                                    "Cannot save mapping because the regenerated cleaned file would be missing "
                                    "required dashboard columns: " + ", ".join(missing)
                                )
                            else:
                                # 2) Save mapping metadata
                                db.upsert_file_column_map(
                                    user["id"],
                                    selected_file_id,
                                    raw_columns=raw_cols,
                                    canon_to_raw=canon_to_raw,
                                    raw_preview_rows=map_rec.get("raw_preview"),
                                )

                                # 3) Overwrite cleaned file bytes in Neon
                                cleaned_bytes2 = df_clean2.to_csv(index=False).encode("utf-8")
                                db.update_file_content(selected_file_id, cleaned_bytes2)

                                # 4) Persist as template for quick upload
                                db.set_last_upload_context(
                                    user["id"],
                                    file_id=selected_file_id,
                                    raw_columns=raw_cols,
                                    canon_to_raw=canon_to_raw,
                                )

                                st.success("Saved mapping selections and regenerated the cleaned file.")
                                st.rerun()

                if map_rec.get("raw_preview"):
                    with st.expander("Raw preview (first 10 rows from the original upload)", expanded=False):
                        raw_prev_df = pd.DataFrame(map_rec["raw_preview"])
                        st.dataframe(with_indexed_headers(raw_prev_df), use_container_width=True)


# ---- Section B: New upload workflow ----
if uploaded is not None:
    st.divider()
    st.subheader("New upload: map columns, generate cleaned file")

    original_name = uploaded.name
    ext = Path(original_name).suffix or ".csv"
    file_bytes_raw = uploaded.getvalue()

    df_raw, file_type, encoding_used, header_row = load_table_from_bytes(file_bytes_raw, ext)
    if header_row > 0:
        st.info(f"Detected headers on row {header_row+1} (skipped {header_row} row(s) above).")
    
    raw_cols = [str(c) for c in df_raw.columns]

    if file_type == "excel":
        st.caption("Read file as Excel (.xlsx/.xls/.xlsm).")
    else:
        st.caption(f"Read CSV using encoding: `{encoding_used}`")

    st.markdown("#### Original file preview (this upload)")
    st.dataframe(with_indexed_headers(df_raw.head(10)), use_container_width=True)


    alias_table = build_alias_table()
    auto_raw_to_canon, _, _ = map_columns(raw_cols, alias_table)

    auto_canon_to_raw = {}
    for raw, canon in auto_raw_to_canon.items():
        auto_canon_to_raw.setdefault(canon, raw)

    default_canon_to_raw = dict(auto_canon_to_raw)
    template = (last_ctx.get("canon_to_raw") or {})
    for canon, raw in template.items():
        if raw and raw in raw_cols:
            default_canon_to_raw[canon] = raw

    st.markdown("#### Step 1: Review and adjust column mapping")
    st.caption(
        "Defaults come from automatic matching, but your most recently saved mapping selections "
        "are used when they match this dataset."
    )

    # ---------- Irrigation UI state (New upload) ----------
    if "newupload_irrig_count" not in st.session_state:
        inferred = 1
        for i in range(1, MAX_IRRIGATION_ZONES + 1):
            if (template or {}).get(f"Irrigation{i}"):
                inferred = i
        st.session_state.newupload_irrig_count = inferred

    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("Add irrigation zone", key="newupload_add_irrig"):
            st.session_state.newupload_irrig_count = min(
                MAX_IRRIGATION_ZONES, st.session_state.newupload_irrig_count + 1
            )
    with b2:
        if st.button("Remove irrigation zone", key="newupload_rm_irrig"):
            st.session_state.newupload_irrig_count = max(1, st.session_state.newupload_irrig_count - 1)

    canon_list_new = CANON_UI_BASE + [f"Irrigation{i}" for i in range(1, st.session_state.newupload_irrig_count + 1)]

    with st.form("new_upload_mapping_form"):
        raw_to_canon, canon_to_raw, duplicates = canon_select_ui(
            raw_cols=raw_cols,
            default_canon_to_raw=default_canon_to_raw,
            form_key="newupload",
            canon_list=canon_list_new,
        )
        submitted = st.form_submit_button("✅ Generate cleaned file and save to Neon")

    if submitted:
        if duplicates:
            st.error(f"Duplicate selections: {', '.join(duplicates)}")
            st.stop()

        if not raw_to_canon:
            st.warning("No columns were mapped. Please select at least one column and try again.")
            st.stop()

        df_clean = build_clean_dataframe(df_raw, raw_to_canon)

        required_for_dashboard = ["Time", "AirTemp", "RH"]
        missing = [c for c in required_for_dashboard if c not in df_clean.columns]
        if missing:
            st.error("Missing required dashboard columns: " + ", ".join(missing))
            st.stop()

        if "Time" in df_clean.columns and not df_clean["Time"].isna().all():
            data_start = df_clean["Time"].min()
            data_start_str = data_start.strftime("%Y%m%dT%H%M")
        else:
            data_start_str = time.strftime("%Y%m%dT%H%M", time.gmtime())

        uname = username_slug(user)

        orig_stem = Path(uploaded.name).stem
        orig_stem = re.sub(r"\s+", "_", orig_stem)
        orig_stem = re.sub(r"[^A-Za-z0-9_-]+", "", orig_stem)
        orig_stem = re.sub(r"_+", "_", orig_stem).strip("_") or "file"

        stored_filename = f"{orig_stem}_{uname}.csv"

        cleaned_bytes = df_clean.to_csv(index=False).encode("utf-8")
        file_id = db.add_file_record(
            user["id"],
            stored_filename,
            cleaned_bytes,
            raw_filename=original_name,
            raw_bytes=file_bytes_raw,
        )

        db.upsert_file_column_map(
            user["id"],
            file_id,
            raw_columns=raw_cols,
            canon_to_raw=canon_to_raw,
            raw_preview_rows=df_raw.head(10).where(pd.notnull(df_raw.head(10)), None).to_dict(orient="records"),
        )

        db.set_last_upload_context(
            user["id"],
            file_id=file_id,
            raw_columns=raw_cols,
            canon_to_raw=canon_to_raw,
        )

        st.success(f"Saved cleaned file in Neon as `{stored_filename}`.")

        st.markdown("#### Cleaned & mapped file preview")
        st.dataframe(df_clean.head(10), use_container_width=True)
        st.caption(f"Detected columns: {', '.join(df_clean.columns)}")

        st.download_button(
            "⬇️ Download cleaned CSV",
            data=cleaned_bytes,
            file_name=stored_filename,
            mime="text/csv",
        )
