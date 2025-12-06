# pages/1_Upload.py

import io
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from lib import auth, db

# ------------- Canonical mapping helpers -------------


def normalize(s: str) -> str:
    """Normalize a column name for matching."""
    s = s.replace("\ufeff", "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)  # "RHT-Temperature" -> "rhttemperature"
    return s


# Canonicals + specific aliases
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


# ------------- Streamlit page -------------


st.set_page_config(page_title="Upload", page_icon="📂", layout="wide")
auth.require_login()
user = auth.current_user()
auth.render_sidebar()

st.title("📂 Upload data file (.csv or .xlsx)")

uploaded = st.file_uploader(
    "Choose a data file",
    type=["csv", "xlsx", "xls"],
)

if uploaded is not None:
    original_name = uploaded.name
    ext = Path(original_name).suffix or ".csv"
    file_bytes_raw = uploaded.getvalue()

    try:
        # Step 1 – load & preview raw file
        df_raw, file_type, encoding_used = load_table_from_bytes(file_bytes_raw, ext)

        if file_type == "excel":
            st.caption("Read file as Excel (.xlsx/.xls).")
        else:
            st.caption(f"Read CSV using encoding: `{encoding_used}`")

        st.subheader("Step 1: Original file preview")
        st.caption("First 10 rows from the uploaded file (raw).")
        st.dataframe(df_raw.head(10), use_container_width=True)

        # Step 2 – automatic mapping & user review
        alias_table = build_alias_table()
        raw_cols = [str(c) for c in df_raw.columns]
        auto_mapping, _, _ = map_columns(raw_cols, alias_table)

        auto_canon_to_raw = {}
        for raw_col, canon in auto_mapping.items():
            auto_canon_to_raw.setdefault(canon, raw_col)

        st.subheader("Step 2: Review and adjust column mapping")
        st.markdown(
            "Select which **raw column** should be used for each canonical field. "
            "Defaults come from automatic alias matching; adjust if needed."
        )

        with st.form("mapping_form"):
            mapping_selections = {}
            options_all = ["(None)"] + raw_cols

            for canon in CANON_ORDER:
                default_raw = auto_canon_to_raw.get(canon)
                if default_raw in raw_cols:
                    default_index = raw_cols.index(default_raw) + 1
                else:
                    default_index = 0

                sel = st.selectbox(
                    f"{canon} column",
                    options_all,
                    index=default_index,
                    key=f"map_{canon}",
                )
                mapping_selections[canon] = sel

            submitted = st.form_submit_button("✅ Accept mapping and generate cleaned file")

        if submitted:
            # Build mapping raw -> canon from user selections
            user_mapping = {}
            used_raw = set()
            duplicate_raws = set()

            for canon in CANON_ORDER:
                raw_sel = mapping_selections.get(canon)
                if not raw_sel or raw_sel == "(None)":
                    continue
                if raw_sel in used_raw:
                    duplicate_raws.add(raw_sel)
                else:
                    used_raw.add(raw_sel)
                    user_mapping[raw_sel] = canon

            if duplicate_raws:
                st.error(
                    "Each raw column can only be mapped to **one** canonical column.\n\n"
                    "Duplicate selections: " + ", ".join(sorted(duplicate_raws))
                )
                st.stop()

            if not user_mapping:
                st.warning(
                    "No columns were mapped. Please select at least one column "
                    "for a canonical field and try again."
                )
                st.stop()

            df_clean = build_clean_dataframe(df_raw, user_mapping)

            # Require at least Time, AirTemp, RH for the dashboard to be useful
            required_for_dashboard = ["Time", "AirTemp", "RH"]
            missing_for_dashboard = [c for c in required_for_dashboard if c not in df_clean.columns]
            if missing_for_dashboard:
                st.error(
                    "The cleaned file is missing required columns for the Dashboard: "
                    + ", ".join(missing_for_dashboard)
                    + ". Please adjust your mapping and try again."
                )
                st.stop()

            # DataStartTime for filename
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
                f"Cleaned file saved in Neon as `{stored_filename}`. "
                "You can now select it on the Dashboard."
            )

            st.subheader("Cleaned & mapped file preview")
            st.caption("First 10 rows from the cleaned CSV (canonical columns).")
            st.dataframe(df_clean.head(10), use_container_width=True)

            st.download_button(
                label="⬇️ Download cleaned CSV",
                data=cleaned_bytes,
                file_name=f"{uname}_{data_start_str}_clean.csv",
                mime="text/csv",
            )

            with st.expander("Final column mapping details", expanded=True):
                rows = [(raw, canon) for raw, canon in user_mapping.items()]
                mapping_df = pd.DataFrame(
                    rows, columns=["Raw column name", "Canonical column"]
                )
                st.table(mapping_df)

    except Exception as e:
        st.error(f"Could not read or process file: {e}")

st.divider()
st.subheader("Your uploads")

files = db.list_user_files(user["id"])
if files:
    for rec in files:
        st.write(f"• {rec['filename']} — uploaded {rec['uploaded_at']}")
else:
    st.caption("No files yet.")
