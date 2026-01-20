# lib/db.py - Postgres/Neon version

from pathlib import Path
from typing import Optional, Dict, Any, List

import streamlit as st
import psycopg2
import psycopg2.extras

import json
import math
from datetime import datetime, date

# We still keep a local data directory for any temporary files if needed
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_conn():
    """
    Get a Postgres connection using the URL in st.secrets['database']['url'].

    We use RealDictCursor so rows behave like dicts (row['column']).
    """
    db_url = st.secrets["database"]["url"]
    conn = psycopg2.connect(
        db_url,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
    return conn


def init_db() -> None:
    """Create tables if they don't exist (idempotent)."""
    conn = get_conn()
    cur = conn.cursor()

    # Users table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id            SERIAL PRIMARY KEY,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin      BOOLEAN DEFAULT FALSE,
            created_at    TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )

    # ---------------- Settings (per user) ----------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            id SERIAL PRIMARY KEY,
            user_id INTEGER UNIQUE NOT NULL REFERENCES users(id),

            -- UNITS
            orig_temp_unit   TEXT DEFAULT 'C',
            orig_light_unit  TEXT DEFAULT 'PPFD',
            temp_unit        TEXT DEFAULT 'F',
            leaf_wetness_unit   TEXT DEFAULT 'Percent',

            -- TARGETS
            target_low       REAL DEFAULT 65.0,
            target_high      REAL DEFAULT 80.0,
            target_rh_low    REAL DEFAULT 70.0,
            target_rh_high   REAL DEFAULT 95.0,
            target_ppfd      REAL DEFAULT 150.0,
            target_dli       REAL DEFAULT 8.0,
            target_vpd_low   REAL DEFAULT 0.2,
            target_vpd_high  REAL DEFAULT 0.8,
            irrigation_trigger          REAL DEFAULT 1.0,
            irrigation_min_interval_min REAL DEFAULT 7.0,
            irrigation_sensitivity_pct  REAL DEFAULT 3.0,
            leaf_wetness_min_interval_min REAL DEFAULT 7.0
        );
        """
    )

    # --- Lightweight migration for older databases ---
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'settings';
        """
    )
    existing_cols = {row["column_name"] for row in cur.fetchall()}

    if "orig_temp_unit" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN orig_temp_unit TEXT DEFAULT 'C';")
    if "orig_light_unit" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN orig_light_unit TEXT DEFAULT 'PPFD';")
    if "temp_unit" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN temp_unit TEXT DEFAULT 'F';")
    if "target_low" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN target_low REAL DEFAULT 65.0;")
    if "target_high" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN target_high REAL DEFAULT 80.0;")
    if "target_rh_low" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN target_rh_low REAL DEFAULT 70.0;")
    if "target_rh_high" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN target_rh_high REAL DEFAULT 95.0;")
    if "target_ppfd" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN target_ppfd REAL DEFAULT 150.0;")
    if "target_dli" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN target_dli REAL DEFAULT 8.0;")
    if "target_vpd_low" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN target_vpd_low REAL DEFAULT 0.2;")
    if "target_vpd_high" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN target_vpd_high REAL DEFAULT 0.8;")
    if "irrigation_trigger" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN irrigation_trigger REAL DEFAULT 1.0;")
    if "irrigation_min_interval_min" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN irrigation_min_interval_min REAL DEFAULT 7.0;")
    if "leaf_wetness_unit" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN leaf_wetness_unit TEXT DEFAULT 'Percent';")
    if "irrigation_sensitivity_pct" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN irrigation_sensitivity_pct REAL DEFAULT 3.0;")
    if "leaf_wetness_min_interval_min" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN leaf_wetness_min_interval_min REAL DEFAULT 7.0;")

    # Persist Upload page state across sessions (last viewed file + mapping template)
    if "last_upload_file_id" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN last_upload_file_id INTEGER;")
    if "last_raw_columns" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN last_raw_columns JSONB;")
    if "last_canon_to_raw" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN last_canon_to_raw JSONB;")
    # Persist Home page state across sessions (last viewed file on Home)
    if "last_home_file_id" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN last_home_file_id INTEGER;")

    # App meta key/value store
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )

    # Files (cleaned uploads stored directly in Neon)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            id          SERIAL PRIMARY KEY,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            filename    TEXT NOT NULL,
            content     BYTEA NOT NULL,
            raw_filename    TEXT,
            raw_content BYTEA,
            uploaded_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )

    # --- Lightweight migration for older databases (files table) ---
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'files';
        """
    )
    files_cols = {row["column_name"] for row in cur.fetchall()}

    if "raw_filename" not in files_cols:
        cur.execute("ALTER TABLE files ADD COLUMN raw_filename TEXT;")
    if "raw_content" not in files_cols:
        cur.execute("ALTER TABLE files ADD COLUMN raw_content BYTEA;")

    # File column mapping + preview metadata (so mapping UI persists across sessions)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS file_column_maps (
            id           SERIAL PRIMARY KEY,
            file_id       INTEGER UNIQUE NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            user_id       INTEGER NOT NULL REFERENCES users(id),
            raw_columns   JSONB,
            canon_to_raw  JSONB,
            raw_preview   JSONB,
            updated_at    TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )

    # --- Lightweight migration for file_column_maps (if table existed before we added columns) ---
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'file_column_maps';
        """
    )
    fcm_cols = {row["column_name"] for row in cur.fetchall()} if cur.rowcount else set()
    if "raw_columns" not in fcm_cols:
        cur.execute("ALTER TABLE file_column_maps ADD COLUMN raw_columns JSONB;")
    if "canon_to_raw" not in fcm_cols:
        cur.execute("ALTER TABLE file_column_maps ADD COLUMN canon_to_raw JSONB;")
    if "raw_preview" not in fcm_cols:
        cur.execute("ALTER TABLE file_column_maps ADD COLUMN raw_preview JSONB;")
    if "updated_at" not in fcm_cols:
        cur.execute("ALTER TABLE file_column_maps ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();")
    
    conn.commit()
    cur.close()
    conn.close()

# -------- Users & settings --------

def get_user_by_username(username: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def add_user(username: str, password_hash: str, is_admin: int = 0) -> int:
    """Insert a user and return its id."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO users (username, password_hash, is_admin)
        VALUES (%s, %s, %s)
        RETURNING id;
        """,
        (username, password_hash, bool(is_admin)),
    )
    user_id = cur.fetchone()["id"]

    # Ensure default settings row exists
    cur.execute(
        "INSERT INTO settings (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING;",
        (user_id,),
    )
    conn.commit()
    cur.close()
    conn.close()
    return user_id


def get_or_create_settings(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO settings (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING;",
        (user_id,),
    )
    conn.commit()
    cur.execute("SELECT * FROM settings WHERE user_id = %s", (user_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def update_settings(
    user_id: int,
    *,
    orig_temp_unit: str,
    orig_light_unit: str,
    temp_unit: str,
    target_low: float,
    target_high: float,
    target_rh_low: float,
    target_rh_high: float,
    target_ppfd: float,
    target_dli: float,
    target_vpd_low: float,
    target_vpd_high: float,
    irrigation_trigger: float,
    irrigation_min_interval_min: float,
    leaf_wetness_unit: str,
    irrigation_sensitivity_pct: float,
    leaf_wetness_min_interval_min: float,
) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE settings
        SET
            orig_temp_unit = %s,
            orig_light_unit = %s,
            temp_unit = %s,
            target_low = %s,
            target_high = %s,
            target_rh_low = %s,
            target_rh_high = %s,
            target_ppfd = %s,
            target_dli = %s,
            target_vpd_low = %s,
            target_vpd_high = %s,
            irrigation_trigger = %s,
            irrigation_min_interval_min = %s,
            leaf_wetness_unit = %s,
            irrigation_sensitivity_pct = %s,
            leaf_wetness_min_interval_min = %s
        WHERE user_id = %s;
        """,
        (
            orig_temp_unit,
            orig_light_unit,
            temp_unit,
            target_low,
            target_high,
            target_rh_low,
            target_rh_high,
            target_ppfd,
            target_dli,
            target_vpd_low,
            target_vpd_high,
            irrigation_trigger,
            irrigation_min_interval_min,
            leaf_wetness_unit,
            irrigation_sensitivity_pct,
            leaf_wetness_min_interval_min,
            user_id,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()


def update_leaf_wetness_event_settings(
    user_id: int,
    *,
    irrigation_sensitivity_pct: float,
    leaf_wetness_min_interval_min: float,
) -> None:
    """
    Update ONLY the Leaf Wetness -> Irrigation Event detection settings.
    Does not touch any other settings fields.
    Uses UPSERT so it works even if the settings row doesn't exist yet.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO settings (user_id, irrigation_sensitivity_pct, leaf_wetness_min_interval_min)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE
        SET
            irrigation_sensitivity_pct = EXCLUDED.irrigation_sensitivity_pct,
            leaf_wetness_min_interval_min = EXCLUDED.leaf_wetness_min_interval_min;
        """,
        (user_id, irrigation_sensitivity_pct, leaf_wetness_min_interval_min),
    )
    conn.commit()
    cur.close()
    conn.close()


# -------- Files (cleaned CSVs in Neon) --------

def list_user_files(user_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, user_id, filename, uploaded_at
        FROM files
        WHERE user_id = %s
        ORDER BY uploaded_at DESC;
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def add_file_record(
    user_id: int,
    filename: str,
    file_bytes: bytes,
    *,
    raw_filename: Optional[str] = None,
    raw_bytes: Optional[bytes] = None,
) -> int:
    """
    Store the CLEANED CSV bytes in Neon and return the created file_id.
    Also stores the original RAW upload bytes so the cleaned file can be regenerated
    later if the user edits the column mapping.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO files (user_id, filename, content, raw_filename, raw_content)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (
            user_id,
            filename,
            psycopg2.Binary(file_bytes),
            raw_filename,
            psycopg2.Binary(raw_bytes) if raw_bytes is not None else None,
        ),
    )
    file_id = cur.fetchone()["id"]
    conn.commit()
    cur.close()
    conn.close()
    return file_id


def get_file_bytes(file_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch a cleaned CSV from Neon.

    Returns {"filename": str, "bytes": bytes} or None.
    Handles BYTEA / memoryview / TEXT just in case.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT filename, content FROM files WHERE id = %s;",
        (file_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return None

    content = row["content"]
    if isinstance(content, memoryview):
        content_bytes = content.tobytes()
    elif isinstance(content, bytes):
        content_bytes = content
    elif isinstance(content, str):
        content_bytes = content.encode("utf-8")
    else:
        raise TypeError(f"Unexpected type for files.content: {type(content)}")

    return {"filename": row["filename"], "bytes": content_bytes}

def get_raw_file_bytes(file_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch the ORIGINAL RAW upload bytes stored for a file.
    Returns {"raw_filename": str, "bytes": bytes} or None if not present.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT raw_filename, raw_content FROM files WHERE id = %s;",
        (file_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row or row.get("raw_content") is None:
        return None

    content = row["raw_content"]
    if isinstance(content, memoryview):
        content_bytes = content.tobytes()
    elif isinstance(content, bytes):
        content_bytes = content
    else:
        raise TypeError(f"Unexpected type for files.raw_content: {type(content)}")

    return {"raw_filename": row.get("raw_filename") or "raw_upload", "bytes": content_bytes}


def update_file_content(file_id: int, file_bytes: bytes) -> None:
    """Overwrite the cleaned CSV bytes stored in Neon for an existing file."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE files SET content = %s WHERE id = %s;",
        (psycopg2.Binary(file_bytes), file_id),
    )
    conn.commit()
    cur.close()
    conn.close()


# -------- Upload mapping persistence --------
def _json_safe(obj):
    """
    Convert Python/Pandas objects into JSON-safe objects:
    - NaN/NaT -> None
    - datetime/date -> ISO string
    - dict/list/tuple -> recurse
    """
    if obj is None:
        return None

    # float('nan')
    if isinstance(obj, float) and math.isnan(obj):
        return None

    # pandas NaT shows up as a NaTType, but often comes through as 'nan' or Timestamp
    if str(obj) == "NaT":
        return None

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    return obj

def upsert_file_column_map(
    user_id: int,
    file_id: int,
    *,
    raw_columns: List[str],
    canon_to_raw: Dict[str, Optional[str]],
    raw_preview_rows: Optional[List[Dict[str, Any]]] = None,
) -> None:
    conn = get_conn()
    cur = conn.cursor()

    safe_raw_columns = _json_safe(raw_columns)
    safe_canon_to_raw = _json_safe(canon_to_raw)
    safe_preview = _json_safe(raw_preview_rows) if raw_preview_rows is not None else None

    cur.execute(
        """
        INSERT INTO file_column_maps (file_id, user_id, raw_columns, canon_to_raw, raw_preview, updated_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (file_id)
        DO UPDATE SET
            raw_columns = EXCLUDED.raw_columns,
            canon_to_raw = EXCLUDED.canon_to_raw,
            raw_preview = EXCLUDED.raw_preview,
            updated_at = NOW();
        """,
        (
            file_id,
            user_id,
            psycopg2.extras.Json(safe_raw_columns),
            psycopg2.extras.Json(safe_canon_to_raw),
            psycopg2.extras.Json(safe_preview) if safe_preview is not None else None,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_file_column_map(file_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT file_id, user_id, raw_columns, canon_to_raw, raw_preview, updated_at
        FROM file_column_maps
        WHERE file_id = %s;
        """,
        (file_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def set_last_upload_context(
    user_id: int,
    *,
    file_id: int,
    raw_columns: List[str],
    canon_to_raw: Dict[str, Optional[str]],
) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO settings (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING;",
        (user_id,),
    )
    cur.execute(
        """
        UPDATE settings
        SET
            last_upload_file_id = %s,
            last_raw_columns = %s,
            last_canon_to_raw = %s
        WHERE user_id = %s;
        """,
        (
            file_id,
            psycopg2.extras.Json(raw_columns),
            psycopg2.extras.Json(canon_to_raw),
            user_id,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_last_upload_context(user_id: int) -> Dict[str, Any]:
    row = get_or_create_settings(user_id)
    if not row:
        return {}
    return {
        "file_id": row.get("last_upload_file_id"),
        "raw_columns": row.get("last_raw_columns") or [],
        "canon_to_raw": row.get("last_canon_to_raw") or {},
    }

def set_last_home_file_id(user_id: int, file_id: Optional[int]) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE settings SET last_home_file_id = %s WHERE user_id = %s;",
        (file_id, user_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_last_home_file_id(user_id: int) -> Optional[int]:
    row = get_or_create_settings(user_id)
    if not row:
        return None
    val = row.get("last_home_file_id")
    return int(val) if val is not None else None


# -------- Meta & admin helpers --------

def get_meta(key: str) -> Optional[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM app_meta WHERE key = %s", (key,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row["value"] if row else None


def set_meta(key: str, value: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO app_meta (key, value)
        VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
        """,
        (key, value),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_user_by_id(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def update_user_password(user_id: int, new_hash: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET password_hash = %s WHERE id = %s;",
        (new_hash, user_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def list_users() -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, username, is_admin, created_at
        FROM users
        ORDER BY id;
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows
