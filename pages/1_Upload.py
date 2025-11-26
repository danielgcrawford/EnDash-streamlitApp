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
    mapping should be {raw_col -> canonical_name}.
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

    # Per-user upload state so we don't keep re-saving the same raw file
    state_key = f"upload_state_{user['id']}"
    state = st.session_state.setdefault(state_key, {})

    # Save raw/original file exactly as uploaded (once per file name)
    if state.get("uploaded_name") != original_name:
        ts = int(time.time())
        raw_path = user_dir / f"{ts}_{original_name}"
        with open(raw_path, "wb") as f:
            f.write(uploaded.getbuffer())
        state["uploaded_name"] = original_name
        state["raw_path"] = str(raw_path)
    raw_path = Path(state["raw_path"])

    try:
        # --- Load raw file ---
        df_raw, file_type, encoding_used = load_table(raw_path)

        if file_type == "excel":
            st.caption("Read file as Excel (.xlsx/.xls).")
        else:
            st.caption(f"Read CSV using encoding: `{encoding_used}`")

        # --- Step 1: Original preview ---
        st.subheader("Step 1: Original file preview")
        st.caption("First 10 rows from the uploaded file (raw).")
        st.dataframe(df_raw.head(10), use_container_width=True)

        # --- Prepare automatic mapping suggestions ---
        alias_table = build_alias_table()
        raw_cols = [str(c) for c in df_raw.columns]
        auto_mapping, auto_missing, auto_extras = map_columns(raw_cols, alias_table)

        # Invert auto_mapping to get default raw per canonical
        auto_canon_to_raw = {}
        for raw_col, canon in auto_mapping.items():
            auto_canon_to_raw.setdefault(canon, raw_col)

        # --- Step 2: Mapping editor ---
        st.subheader("Step 2: Review and adjust column mapping")
        st.markdown(
            "Select which **raw column** should be used for each canonical field. "
            "Defaults are based on automatic alias matching, but you can adjust them "
            "if a column was mis-detected or has an unusual name."
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

                selection = st.selectbox(
                    f"{canon} column",
                    options_all,
                    index=default_index,
                    key=f"map_{canon}",
                    help=f"Select which column from your file should be treated as `{canon}`."
                )
                mapping_selections[canon] = selection

            st.caption(
                "Columns left as **(None)** will not appear in the cleaned file. "
                "Each raw column should be mapped to at most one canonical name."
            )
            submitted = st.form_submit_button("✅ Accept mapping and generate cleaned file")

        # --- Step 3: Generate & preview cleaned file after confirmation ---
        if submitted:
            # Build mapping {raw -> canon} from form selections
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
            elif not user_mapping:
                st.warning(
                    "No columns were mapped. Please select at least one column "
                    "for a canonical field and try again."
                )
            else:
                df_clean = build_clean_dataframe(df_raw, user_mapping)

                clean_ts = int(time.time())
                clean_filename = f"{clean_ts}_{Path(original_name).stem}_clean.csv"
                clean_path = user_dir / clean_filename

                df_clean.to_csv(clean_path, index=False)

                # Record *clean* file in DB so the Dashboard uses mapped data
                db.add_file_record(user["id"], original_name, str(clean_path))

                st.success(
                    f"Cleaned file generated and saved as `{clean_filename}`. "
                    "This file will appear in the Dashboard file selector."
                )

                st.subheader("Cleaned & mapped file preview")
                if df_clean.empty:
                    st.warning(
                        "The cleaned file is empty after applying the selected mapping. "
                        "Check that your chosen columns contain valid data."
                    )
                else:
                    st.caption("First 10 rows from the cleaned CSV (canonical columns).")
                    st.dataframe(df_clean.head(10), use_container_width=True)

                    # Download cleaned CSV
                    st.download_button(
                        label="⬇️ Download cleaned CSV",
                        data=df_clean.to_csv(index=False).encode("utf-8"),
                        file_name=clean_filename,
                        mime="text/csv",
                    )

                # Final mapping details
                with st.expander("Final column mapping details", expanded=True):
                    mapping_rows = [(raw, canon) for raw, canon in user_mapping.items()]
                    mapping_df = pd.DataFrame(
                        mapping_rows,
                        columns=["Raw column name", "Canonical column"],
                    )
                    st.table(mapping_df)

                    missing_manual = [
                        canon for canon in CANON_ORDER
                        if canon not in user_mapping.values()
                    ]
                    if missing_manual:
                        st.write("**Canonical columns not included in cleaned file:**")
                        st.write(", ".join(missing_manual))

                    extras_manual = [
                        c for c in raw_cols if c not in user_mapping
                    ]
                    if extras_manual:
                        st.write("**Unmapped raw columns (remain only in original file):**")
                        st.write(", ".join(extras_manual))

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

