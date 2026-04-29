"""
db.py — Database setup and helpers.
SQLite stores everything in a single file (attendance.db) — no server needed.
"""

import sqlite3
import numpy as np
from datetime import datetime, date
from pathlib import Path

DB_PATH = Path(__file__).parent / "attendance.db"


def get_conn():
    """Open a connection. Row factory lets us access columns by name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call repeatedly."""
    conn = get_conn()
    cur = conn.cursor()

    # One row per enrolled child.
    # face_encoding is a 512-float ArcFace embedding stored as raw bytes.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            class_name TEXT NOT NULL,
            face_encoding BLOB NOT NULL,
            audio_path TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # One row per (student, date). UNIQUE prevents duplicate marking.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            arrival_time TEXT,
            status TEXT NOT NULL DEFAULT 'present',
            manually_edited INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (student_id) REFERENCES students(id),
            UNIQUE (student_id, date)
        )
    """)

    conn.commit()
    conn.close()


def encoding_to_bytes(encoding: np.ndarray) -> bytes:
    """Convert a 128-D numpy face encoding to bytes for storage."""
    return encoding.astype(np.float64).tobytes()


def bytes_to_encoding(data: bytes) -> np.ndarray:
    """Convert stored bytes back into a numpy encoding."""
    return np.frombuffer(data, dtype=np.float64)


def add_student(name, class_name, face_encoding, audio_path):
    """Insert a new student. face_encoding is a numpy array."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO students (name, class_name, face_encoding, audio_path, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (name, class_name, encoding_to_bytes(face_encoding),
         str(audio_path), datetime.now().isoformat())
    )
    conn.commit()
    student_id = cur.lastrowid
    conn.close()
    return student_id


def get_all_students():
    """Return every enrolled student as a list of dicts (with decoded encodings)."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM students ORDER BY class_name, name").fetchall()
    conn.close()
    students = []
    for r in rows:
        d = dict(r)
        d["face_encoding"] = bytes_to_encoding(d["face_encoding"])
        students.append(d)
    return students


def get_student(student_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["face_encoding"] = bytes_to_encoding(d["face_encoding"])
    return d


def delete_student(student_id):
    conn = get_conn()
    conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.close()


def mark_present(student_id, when=None):
    """
    Mark a student present for today. Idempotent — if they're already marked,
    does nothing and returns False. Returns True if a new record was created.
    """
    if when is None:
        when = datetime.now()
    today = when.date().isoformat()
    arrival = when.strftime("%H:%M:%S")

    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO attendance (student_id, date, arrival_time, status) "
            "VALUES (?, ?, ?, 'present')",
            (student_id, today, arrival)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Already marked for today — that's fine, just skip.
        return False
    finally:
        conn.close()


def is_marked_today(student_id, on_date=None):
    """Has this student already been marked for the given date (default: today)?"""
    if on_date is None:
        on_date = date.today().isoformat()
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM attendance WHERE student_id = ? AND date = ?",
        (student_id, on_date)
    ).fetchone()
    conn.close()
    return row is not None


def get_attendance_for_date(on_date=None):
    """
    Return every enrolled student with their attendance status for the given date.
    Status will be 'present', 'absent', or 'not_marked' (no record yet).
    """
    if on_date is None:
        on_date = date.today().isoformat()
    conn = get_conn()
    rows = conn.execute("""
        SELECT s.id, s.name, s.class_name,
               a.arrival_time, a.status, a.manually_edited
        FROM students s
        LEFT JOIN attendance a
          ON a.student_id = s.id AND a.date = ?
        ORDER BY s.class_name, s.name
    """, (on_date,)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d["status"] is None:
            d["status"] = "not_marked"
        result.append(d)
    return result


def set_attendance_status(student_id, status, on_date=None):
    """Manually set status ('present' / 'absent') for a student on a date."""
    if on_date is None:
        on_date = date.today().isoformat()
    if status not in ("present", "absent"):
        raise ValueError("status must be 'present' or 'absent'")

    conn = get_conn()
    arrival = datetime.now().strftime("%H:%M:%S") if status == "present" else None

    # Upsert: try insert, on conflict update.
    conn.execute("""
        INSERT INTO attendance (student_id, date, arrival_time, status, manually_edited)
        VALUES (?, ?, ?, ?, 1)
        ON CONFLICT(student_id, date) DO UPDATE SET
          status = excluded.status,
          arrival_time = COALESCE(excluded.arrival_time, attendance.arrival_time),
          manually_edited = 1
    """, (student_id, on_date, arrival, status))
    conn.commit()
    conn.close()


def flag_absences(on_date=None):
    """
    For every enrolled student who has NO record on the given date,
    insert an 'absent' record. Run this at the cutoff time (e.g. 9:30 AM).
    """
    if on_date is None:
        on_date = date.today().isoformat()
    conn = get_conn()
    conn.execute("""
        INSERT INTO attendance (student_id, date, status)
        SELECT s.id, ?, 'absent'
        FROM students s
        WHERE NOT EXISTS (
          SELECT 1 FROM attendance a
          WHERE a.student_id = s.id AND a.date = ?
        )
    """, (on_date, on_date))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Run `python db.py` once to create the database file.
    init_db()
    print(f"Database ready at: {DB_PATH}")
