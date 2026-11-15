"""Database layer for Physiotherapy Clinic Chatbot.

Provides SQLite initialization, schema definition, and helper functions for
managing FAQs, chat sessions, and message logs.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "chatbot.db"
DEFAULT_FAQS_PATH = DEFAULT_DATA_DIR / "faqs.json"


def get_db_path(custom_path: Optional[str | Path] = None) -> Path:
    """Resolve database path, ensuring the parent directory exists."""
    path = Path(custom_path) if custom_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """Create and return a configured SQLite connection."""
    target_path = get_db_path(db_path)
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(
    db_path: Optional[str | Path] = None,
    faqs_json_path: Optional[str | Path] = None,
) -> None:
    """Initialize database tables and seed FAQs from data/faqs.json if available.

    Tables created:
    - faqs: id, category, question, alt_questions, answer
    - sessions: id, created_at, last_category
    - messages: id, session_id, role, text, matched_faq_id, confidence, flag, timestamp
    """
    target_path = get_db_path(db_path)
    conn = get_connection(target_path)

    with conn:
        # 1. faqs table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faqs (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                question TEXT NOT NULL,
                alt_questions TEXT NOT NULL,
                answer TEXT NOT NULL,
                answers TEXT
            );
            """
        )

        # Migration: ensure answers column exists if table already existed
        try:
            conn.execute("ALTER TABLE faqs ADD COLUMN answers TEXT;")
        except sqlite3.OperationalError:
            pass

        # 2. sessions table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_category TEXT
            );
            """
        )

        # 3. messages table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                matched_faq_id TEXT,
                confidence REAL,
                flag TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            );
            """
        )

        # Create indexes for fast querying
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id DESC);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages (timestamp DESC);"
        )

    # Seed FAQs from JSON file if provided/available
    seed_path = Path(faqs_json_path) if faqs_json_path else DEFAULT_FAQS_PATH
    if seed_path.exists():
        seed_faqs_from_file(seed_path, db_path=target_path)

    conn.close()


def seed_faqs_from_file(
    faqs_json_path: Path, db_path: Optional[str | Path] = None
) -> int:
    """Load and upsert FAQs from a JSON file into the database."""
    with open(faqs_json_path, "r", encoding="utf-8") as f:
        faqs_data = json.load(f)

    conn = get_connection(db_path)
    inserted_count = 0

    with conn:
        for faq in faqs_data:
            faq_id = str(faq.get("id"))
            category = faq.get("category", "General")
            question = faq.get("question", "").strip()
            alt_questions = faq.get("alt_questions", [])
            alt_questions_json = json.dumps(alt_questions, ensure_ascii=False)
            answer = faq.get("answer", "").strip()
            answers = faq.get("answers", [])
            if not answers and answer:
                answers = [answer]
            answers_json = json.dumps(answers, ensure_ascii=False)

            conn.execute(
                """
                INSERT INTO faqs (id, category, question, alt_questions, answer, answers)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    category = excluded.category,
                    question = excluded.question,
                    alt_questions = excluded.alt_questions,
                    answer = excluded.answer,
                    answers = excluded.answers;
                """,
                (faq_id, category, question, alt_questions_json, answer, answers_json),
            )
            inserted_count += 1

    conn.close()
    return inserted_count


def get_all_faqs(db_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all FAQs stored in the database."""
    conn = get_connection(db_path)
    cursor = conn.execute("SELECT * FROM faqs;")
    rows = cursor.fetchall()
    conn.close()

    faqs = []
    for r in rows:
        col_names = r.keys()
        try:
            alt_q = json.loads(r["alt_questions"])
        except Exception:
            alt_q = []

        answers_list: List[str] = []
        if "answers" in col_names and r["answers"]:
            try:
                answers_list = json.loads(r["answers"])
            except Exception:
                answers_list = []
        if not answers_list and r["answer"]:
            answers_list = [r["answer"]]

        faqs.append(
            {
                "id": r["id"],
                "category": r["category"],
                "question": r["question"],
                "alt_questions": alt_q,
                "answer": r["answer"],
                "answers": answers_list,
            }
        )
    return faqs


def get_faq_by_id(faq_id: str, db_path: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single FAQ by its ID."""
    conn = get_connection(db_path)
    cursor = conn.execute(
        "SELECT * FROM faqs WHERE id = ?;",
        (faq_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    col_names = row.keys()
    try:
        alt_q = json.loads(row["alt_questions"])
    except Exception:
        alt_q = []

    answers_list: List[str] = []
    if "answers" in col_names and row["answers"]:
        try:
            answers_list = json.loads(row["answers"])
        except Exception:
            answers_list = []
    if not answers_list and row["answer"]:
        answers_list = [row["answer"]]

    return {
        "id": row["id"],
        "category": row["category"],
        "question": row["question"],
        "alt_questions": alt_q,
        "answer": row["answer"],
        "answers": answers_list,
    }


def get_or_create_session(
    session_id: str, db_path: Optional[str | Path] = None
) -> Dict[str, Any]:
    """Retrieve an existing session or create a new session if not found."""
    conn = get_connection(db_path)
    cursor = conn.execute(
        "SELECT id, created_at, last_category FROM sessions WHERE id = ?;",
        (session_id,),
    )
    row = cursor.fetchone()

    if row:
        session = {
            "id": row["id"],
            "created_at": row["created_at"],
            "last_category": row["last_category"],
        }
        conn.close()
        return session

    # Create new session
    now_iso = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO sessions (id, created_at, last_category)
            VALUES (?, ?, NULL);
            """,
            (session_id, now_iso),
        )

    conn.close()
    return {"id": session_id, "created_at": now_iso, "last_category": None}


def get_session(session_id: str, db_path: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    """Get session details by ID."""
    conn = get_connection(db_path)
    cursor = conn.execute(
        "SELECT id, created_at, last_category FROM sessions WHERE id = ?;",
        (session_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "last_category": row["last_category"],
    }


def update_session_category(
    session_id: str, last_category: Optional[str], db_path: Optional[str | Path] = None
) -> None:
    """Update the last_category for a given session."""
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "UPDATE sessions SET last_category = ? WHERE id = ?;",
            (last_category, session_id),
        )
    conn.close()


def log_message(
    session_id: str,
    role: str,
    text: str,
    matched_faq_id: Optional[str] = None,
    confidence: Optional[float] = None,
    flag: Optional[str] = None,
    timestamp: Optional[str] = None,
    db_path: Optional[str | Path] = None,
) -> int:
    """Log a user or assistant exchange into the messages table."""
    conn = get_connection(db_path)
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    with conn:
        cursor = conn.execute(
            """
            INSERT INTO messages (
                session_id, role, text, matched_faq_id, confidence, flag, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (session_id, role, text, matched_faq_id, confidence, flag, ts),
        )
        msg_id = cursor.lastrowid or 0

    conn.close()
    return msg_id


def get_recent_messages(
    session_id: str, limit: int = 3, db_path: Optional[str | Path] = None
) -> List[Dict[str, Any]]:
    """Retrieve the most recent messages for a session in chronological order."""
    conn = get_connection(db_path)
    cursor = conn.execute(
        """
        SELECT id, session_id, role, text, matched_faq_id, confidence, flag, timestamp
        FROM messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?;
        """,
        (session_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()

    # Reverse so the returned list is in chronological order (oldest to newest among recent)
    messages = [
        {
            "id": r["id"],
            "session_id": r["session_id"],
            "role": r["role"],
            "text": r["text"],
            "matched_faq_id": r["matched_faq_id"],
            "confidence": r["confidence"],
            "flag": r["flag"],
            "timestamp": r["timestamp"],
        }
        for r in reversed(rows)
    ]
    return messages


def get_logs(
    session_id: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve conversation logs for auditing or the GET /logs endpoint."""
    conn = get_connection(db_path)
    if session_id:
        cursor = conn.execute(
            """
            SELECT id, session_id, role, text, matched_faq_id, confidence, flag, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?;
            """,
            (session_id, limit),
        )
    else:
        cursor = conn.execute(
            """
            SELECT id, session_id, role, text, matched_faq_id, confidence, flag, timestamp
            FROM messages
            ORDER BY id DESC
            LIMIT ?;
            """,
            (limit,),
        )
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r["id"],
            "session_id": r["session_id"],
            "role": r["role"],
            "text": r["text"],
            "matched_faq_id": r["matched_faq_id"],
            "confidence": r["confidence"],
            "flag": r["flag"],
            "timestamp": r["timestamp"],
        }
        for r in rows
    ]
