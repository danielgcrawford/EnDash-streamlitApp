# pages/1_Upload.py

import io
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from lib import auth, db


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
}

MAX_IRRIGATION_ZONES = 5

CANON_BASE = ["Time", "AirTemp", "LeafTemp", "RH", "PAR","LeafWetness"]
IRR_CANONS = [f"Irrigation{i}" for i in range(1, MAX_IRRIGATION_ZONES + 1)]
CANON_ORDER = CANON_BASE + IRR_CANONS  # full possible export order

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


def load_table_from_bytes(file_bytes: bytes, ext: str) -> Tuple[pd.DataFrame, str, Optional[str]]:
    ext = ext.lower()

    if ext in [".xlsx", ".xls", ".xlsm"]:
        df = pd.read_excel(io.BytesIO(file_bytes))
        return df, "excel", None

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            return df, "csv", enc
        except Exception as e:
            last_err = e

    raise last_err if last_err is not None else ValueError("Could not read file.")


def build_clean_dataframe(df_raw: pd.DataFrame, raw_to_canon: Dict[str, str]) -> pd.DataFrame:
    canon_to_raw: Dict[str, str] = {}
    for raw, canon in raw_to_canon.items():
        canon_to_raw.setdefault(canon, raw)

    #only export irrigaiton canons that are actually mapped
    canon_order = CANON_BASE + [
        f"irrigation{i}"
        for i in range(1, MAX_IRRIGATION_ZONES + 1)
        if f"Irrigation{i}" in canon_to_raw
    ]

    data = {}
    for canon in CANON_ORDER:
        raw = canon_to_raw.get(canon)
        if raw is None or raw not in df_raw.columns:
            continue

        s = df_raw[raw]
        if canon == "Time":
            s = pd.to_datetime(s, errors="coerce")
        else:
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

uploaded = st.file_uploader("Choose a data file", type=["csv", "xlsx", "xls", "xlsm"])
new_upload_active = uploaded is not None

# ---- Previously uploaded file selector (persistent) ----
st.caption("Select a previously uploaded cleaned file to review/download it and manage its saved column mapping.")
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
        "Previously uploaded files",
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

                canon_list_saved = CANON_BASE + [f"Irrigation{i}" for i in range(1, st.session_state.savedmap_irrig_count + 1)]

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
                        st.success("Saved. These selections will be used as defaults for Quick Upload.")

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

    df_raw, file_type, encoding_used = load_table_from_bytes(file_bytes_raw, ext)
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

    canon_list_new = CANON_BASE + [f"Irrigation{i}" for i in range(1, st.session_state.newupload_irrig_count + 1)]

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
        file_id = db.add_file_record(user["id"], stored_filename, cleaned_bytes)

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
