# lib/db.py
from pathlib import Path
import sqlite3
from typing import Optional

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "app.db"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_conn() -> sqlite3.Connection:
    """Get a SQLite connection. check_same_thread=False for Streamlit threads."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Create tables if they don't exist."""
    conn = get_conn()
    cur = conn.cursor()
    # Users
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    #Updated settings table 11/19
    # Settings (one row per user) 
    # NOTE: Older versions of the app may have created this table without
    # target_low / target_high, so we also run a small migration below.
    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE NOT NULL,
        temp_unit TEXT DEFAULT 'F',           -- 'C' | 'F'
        target_low REAL DEFAULT 65.0,         -- new: low threshold
        target_high REAL DEFAULT 80.0,        -- new: high threshold
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)

    #New Settings storage 11/19/25
    # --- Lightweight migration for existing databases ---
    # If the table already existed without target_low / target_high, add them.
    cur.execute("PRAGMA table_info(settings);")
    existing_cols = [row["name"] for row in cur.fetchall()]

    if "target_low" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN target_low REAL DEFAULT 65.0;")
    if "target_high" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN target_high REAL DEFAULT 80.0;")
    if "temp_unit" not in existing_cols:
        cur.execute("ALTER TABLE settings ADD COLUMN temp_unit TEXT DEFAULT 'F';")
    #New Execute App Meta 11/18/25
    cur.execute("""
    CREATE TABLE IF NOT EXISTS app_meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    """)
    # Files (uploads)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        saved_path TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)
    conn.commit()

def get_user_by_username(username: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    return cur.fetchone()

def add_user(username: str, password_hash: str, is_admin: int = 0) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users(username, password_hash, is_admin) VALUES (?,?,?)",
        (username, password_hash, is_admin),
    )
    user_id = cur.lastrowid
    # Ensure default settings row
    cur.execute("INSERT OR IGNORE INTO settings(user_id) VALUES (?)", (user_id,))
    conn.commit()
    return user_id

def get_or_create_settings(user_id: int) -> sqlite3.Row:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO settings(user_id) VALUES (?)", (user_id,))
    conn.commit()
    cur.execute("SELECT * FROM settings WHERE user_id = ?", (user_id,))
    return cur.fetchone()

#Updated 11/19/25 new settings page
def update_settings(user_id: int, temp_unit: str,
                    target_low: float, target_high: float) -> None:
    """Update per-user temperature settings.

    Called by the Settings page with:
        db.update_settings(user["id"], selected_unit, low_input, high_input)
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE settings
        SET temp_unit = ?, target_low = ?, target_high = ?
        WHERE user_id = ?
    """, (temp_unit, target_low, target_high, user_id))
    conn.commit()

def list_user_files(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM files WHERE user_id=? ORDER BY uploaded_at DESC", (user_id,))
    return cur.fetchall()

def add_file_record(user_id: int, filename: str, saved_path: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO files(user_id, filename, saved_path) VALUES (?,?,?)",
        (user_id, filename, saved_path),
    )
    conn.commit()
#New App Meta - 11/18/25
def get_meta(key: str) -> Optional[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM app_meta WHERE key=?", (key,))
    row = cur.fetchone()
    return row["value"] if row else None

def set_meta(key: str, value: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO app_meta(key, value) VALUES(?,?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
    """, (key, value))
    conn.commit()
#New Users - 11/18/25
def get_user_by_id(user_id: int) -> Optional[sqlite3.Row]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cur.fetchone()

def update_user_password(user_id: int, new_hash: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
    conn.commit()
