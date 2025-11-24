# lib/db.py - Postgres/Neon version
from pathlib import Path
from typing import Optional

import streamlit as st
import psycopg2
import psycopg2.extras

# Paths (still used for uploads etc.)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_conn():
    """
    Get a Postgres connection using the URL in st.secrets['database']['url'].

    We use RealDictCursor so rows behave like sqlite3.Row (row['column']).
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

    # Settings table (one row per user)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            id          SERIAL PRIMARY KEY,
            user_id     INTEGER UNIQUE NOT NULL REFERENCES users(id),
            temp_unit   TEXT DEFAULT 'F',
            target_low  DOUBLE PRECISION DEFAULT 65.0,
            target_high DOUBLE PRECISION DEFAULT 80.0
        );
        """
    )

    # App meta key/value store
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )

    # Files (uploads)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            id          SERIAL PRIMARY KEY,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            filename    TEXT NOT NULL,
            saved_path  TEXT NOT NULL,
            uploaded_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )

    conn.commit()
    cur.close()
    conn.close()


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


def update_settings(user_id: int, temp_unit: str, target_low: float, target_high: float) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE settings
        SET temp_unit = %s,
            target_low = %s,
            target_high = %s
        WHERE user_id = %s;
        """,
        (temp_unit, target_low, target_high, user_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def list_user_files(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM files WHERE user_id = %s ORDER BY uploaded_at DESC;",
        (user_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def add_file_record(user_id: int, filename: str, saved_path: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO files (user_id, filename, saved_path)
        VALUES (%s, %s, %s);
        """,
        (user_id, filename, saved_path),
    )
    conn.commit()
    cur.close()
    conn.close()


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

def list_users():
    """Return basic info for all users."""
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
