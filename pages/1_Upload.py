# pages/1_Upload.py
import streamlit as st
import pandas as pd
from pathlib import Path
import time
import re

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
        "time", "timestamp", "date_time", "datetime", "recorded at",
        "date.time", "logtime", "measurement_time"
    ],
    "AirTemp": [
        "airtemp", "air_temp", "tair", "t_air", "ambient_temp",
        "air temperature", "air temperature (c)", "ta_c",
        "rhttemperature", "RHT - Temperature", "rhttemp", "RHT-Temperature"
    ],
    "LeafTemp": [
        "leaftemp", "leaf_temp", "tleaf", "leaf temperature",
        "canopy_temp", "tc_leaf", "leaf_t (c)", "leaf_tc"
    ],
    "RH": [
        "rel_hum", "relative_humidity", "humidity", "rh (%)",
        "rhhumidity", "rht_humidity", "rh_percent"
    ],
    "PAR": [
        "par", "ppfd", "photosynthetically active radiation", "par_umol",
        "par (umol m-2 s-1)", "par_umolm2s", "quantum",
        "quantum_sensor", "quantumsensor", "quantumpar"
    ],
}
CANON_ORDER = ["Time", "AirTemp", "LeafTemp", "RH", "PAR"]


def build_alias_table():
    """Build a lookup of canonical -> set of normalized aliases (incl. canonical)."""
    table = {}
    for canon, aliases in ALIASES.items():
        table[canon] = {normalize(canon), *[normalize(a) for a in aliases]}
    return table


def map_columns(raw_cols, alias_table):
    """
    Given raw column names and an alias table:
      - Return mapping {raw_name -> canonical_name}
      - List of missing canonicals
      - List of unmapped raw columns (kept in raw, excluded from clean)
    """
    norm_to_raw = {normalize(c): c for c in raw_cols}
    mapping, used = {}, set()

    # Exact/alias matches
    for canon, norms in alias_table.items():
        for norm, raw in norm_to_raw.items():
            if norm in norms and raw not in used:
                mapping[raw] = canon
                used.add(raw)
                break

    # Simple fuzzy "contains" fallback
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
    extras = [c for c in raw_cols if c not in mapping]  # unmapped raw cols
    return mapping, missing, extras


def load_table(path: Path):
    """
    Load a CSV or Excel file.
    Returns (df, file_type, encoding_used)
    file_type in {"csv", "excel"}; encoding_used is None for Excel.
    """
    suffix = path.suffix.lower()

    # Excel
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
        return df, "excel", None

    # CSV with encoding fallback
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, "csv", enc
        except Exception as e:
            last_err = e

    if last_err is not None:
        raise last_err
    raise ValueError("Could not read file with any of the tried encodings.")


def build_clean_dataframe(df_raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Create a clean DataFrame containing only mapped canonical columns,
    in a consistent order (CANON_ORDER).
    """
    # Map each canonical to the first raw column that was mapped to it
    canon_to_raw = {}
    for raw, canon in mapping.items():
        canon_to_raw.setdefault(canon, raw)

    data = {}
    # Build clean columns in canonical order
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
        # No mapped columns – return empty DF
        return pd.DataFrame()

    df_clean = pd.DataFrame(data)

    # Drop rows with invalid Time and sort, if present
    if "Time" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["Time"]).sort_values("Time")

    # Drop rows that are entirely NaN and duplicates
    df_clean = df_clean.dropna(axis=0, how="all").drop_duplicates()

    return df_clean


# ------------- Streamlit page -------------

st.set_page_config(page_title="Upload", page_icon="📂", layout="wide")
auth.require_login()
user = auth.current_user()

st.title("📂 Upload data file (.csv or .xlsx)")

uploaded = st.file_uploader(
    "Choose a data file",
    type=["csv", "xlsx", "xls"]
)

if uploaded is not None:
    original_name = Path(uploaded.name).name
    user_dir = Path("data") / "uploads" / str(user["id"])
    user_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())

    # Save raw/original file exactly as uploaded
    raw_path = user_dir / f"{ts}_{original_name}"
    with open(raw_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Clean file name for the processed CSV
    clean_filename = f"{ts}_{Path(original_name).stem}_clean.csv"
    clean_path = user_dir / clean_filename

    try:
        # --- Load raw file ---
        df_raw, file_type, encoding_used = load_table(raw_path)

        if file_type == "excel":
            st.caption("Read file as Excel (.xlsx/.xls).")
        else:
            st.caption(f"Read CSV using encoding: `{encoding_used}`")

        # --- Build mapping & clean DF ---
        alias_table = build_alias_table()
        raw_cols = [str(c) for c in df_raw.columns]
        mapping, missing, extras = map_columns(raw_cols, alias_table)

        df_clean = build_clean_dataframe(df_raw, mapping)

        # --- Save cleaned file as CSV ---
        df_clean.to_csv(clean_path, index=False)

        # Record *clean* file in DB so the Dashboard uses mapped data
        db.add_file_record(user["id"], original_name, str(clean_path))
        st.success(
            f"Uploaded `{original_name}` and created cleaned file `{clean_filename}`."
        )

        # --- Show previews ---
        st.subheader("Original file preview")
        st.caption("First 10 rows from the uploaded file (raw).")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.subheader("Cleaned & mapped file preview")
        if df_clean.empty:
            st.warning(
                "No columns could be mapped to canonical names. "
                "Check that your column headers match expected aliases."
            )
        else:
            st.caption("First 10 rows from the cleaned CSV (mapped canonical columns).")
            st.dataframe(df_clean.head(10), use_container_width=True)

            # Download cleaned CSV
            st.download_button(
                label="⬇️ Download cleaned CSV",
                data=df_clean.to_csv(index=False).encode("utf-8"),
                file_name=clean_filename,
                mime="text/csv",
            )

        # --- Mapping details ---
        with st.expander("Column mapping details"):
            st.write("**Raw → canonical mapping**")
            if mapping:
                mapping_df = pd.DataFrame(
                    [(raw, canon) for raw, canon in mapping.items()],
                    columns=["Raw column name", "Canonical column"]
                )
                st.table(mapping_df)
            else:
                st.caption("No known canonical columns were detected.")

            if missing:
                st.write("**Canonical columns not found in your file:**")
                st.write(", ".join(missing))

            if extras:
                st.write("**Unmapped columns in original file (not included in clean CSV):**")
                st.write(", ".join(extras))

    except Exception as e:
        st.error(f"Could not read or process file: {e}")

st.divider()
st.subheader("Your uploads")

files = db.list_user_files(user["id"])
if files:
    for rec in files:
        st.write(f"• {rec['filename']}  — uploaded {rec['uploaded_at']}")
else:
    st.caption("No files yet.")
