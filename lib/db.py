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
    # Settings (one row per user)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE NOT NULL,
        unit_pref TEXT DEFAULT 'metric',      -- 'metric' | 'imperial'
        temp_unit TEXT DEFAULT 'C',           -- 'C' | 'F'
        temp_setpoint REAL DEFAULT 24.0,
        rh_setpoint REAL DEFAULT 70.0,
        vpd_setpoint REAL DEFAULT 0.8,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)
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

def update_settings(user_id: int, unit_pref: str, temp_unit: str,
                    temp_setpoint: float, rh_setpoint: float, vpd_setpoint: float) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE settings
        SET unit_pref=?, temp_unit=?, temp_setpoint=?, rh_setpoint=?, vpd_setpoint=?
        WHERE user_id=?
    """, (unit_pref, temp_unit, temp_setpoint, rh_setpoint, vpd_setpoint, user_id))
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
