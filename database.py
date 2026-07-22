"""
Database configuration and setup for Deepfake Detection System.
Supports both MySQL and SQLite (fallback if MySQL is not available).
"""

import sqlite3
import os
import json
from datetime import datetime

# ─── Try MySQL first, fallback to SQLite ───────────────────────────────────────
USE_MYSQL = False
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",          # ← Change to your MySQL password
    "database": "deepfake_db",
    "port": 3306,
}

SQLITE_PATH = os.path.join(os.path.dirname(__file__), "deepfake_detections.db")


def get_connection():
    """Get a database connection (MySQL or SQLite)."""
    if USE_MYSQL:
        try:
            import mysql.connector
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            return conn, "mysql"
        except Exception as e:
            print(f"[DB] MySQL unavailable ({e}), falling back to SQLite.")
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"


def init_db():
    """Create tables if they don't exist."""
    conn, db_type = get_connection()
    cursor = conn.cursor()

    if db_type == "mysql":
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(20) NOT NULL,
                verdict VARCHAR(20) NOT NULL,
                confidence FLOAT NOT NULL,
                details TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
    else:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                details TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)

    conn.commit()
    conn.close()
    print(f"[DB] Initialized ({db_type})")


def save_detection(filename, file_type, verdict, confidence, details: dict):
    """Persist a detection result to the database."""
    conn, db_type = get_connection()
    cursor = conn.cursor()

    details_json = json.dumps(details)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if db_type == "mysql":
        cursor.execute(
            "INSERT INTO detections (filename, file_type, verdict, confidence, details, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
            (filename, file_type, verdict, confidence, details_json, now),
        )
    else:
        cursor.execute(
            "INSERT INTO detections (filename, file_type, verdict, confidence, details, created_at) VALUES (?,?,?,?,?,?)",
            (filename, file_type, verdict, confidence, details_json, now),
        )

    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id


def get_recent_detections(limit=20):
    """Fetch the latest detection records."""
    conn, db_type = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
        if db_type == "sqlite"
        else (limit,),
    )
    # MySQL uses %s-style but LIMIT ? still works for both in this context
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        r = dict(row)
        try:
            r["details"] = json.loads(r.get("details") or "{}")
        except Exception:
            r["details"] = {}
        results.append(r)
    return results
