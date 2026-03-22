import json
import os
import sqlite3
from datetime import datetime
from threading import Lock
from typing import Any


class TicketStore:
    def __init__(self, db_path: str, max_rows: int = 150) -> None:
        self.db_path = db_path
        self.max_rows = max_rows
        self._lock = Lock()

    def initialize(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    language TEXT NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    cleaned_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    processing_timeline_json TEXT NOT NULL,
                    total_processing_ms INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ticket_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (ticket_id) REFERENCES tickets(id)
                )
                """
            )

    def next_counter(self, default_counter: int = 1200) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(CAST(SUBSTR(id, 4) AS INTEGER))
                FROM tickets
                WHERE id LIKE 'GT-%'
                """
            ).fetchone()

        max_id = row[0] if row and row[0] is not None else None
        return max(max_id or default_counter, default_counter)

    def insert_ticket(self, ticket: dict[str, Any]) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO tickets (
                        id,
                        channel,
                        language,
                        category,
                        status,
                        priority,
                        cleaned_text,
                        confidence,
                        processing_timeline_json,
                        total_processing_ms,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticket["id"],
                        ticket["channel"],
                        ticket["language"],
                        ticket["category"],
                        ticket["status"],
                        ticket["priority"],
                        ticket["cleaned_text"],
                        ticket["confidence"],
                        json.dumps(ticket["processing_timeline"]),
                        ticket["total_processing_ms"],
                        ticket["created_at"],
                    ),
                )
                conn.execute(
                    """
                    DELETE FROM tickets
                    WHERE id NOT IN (
                        SELECT id FROM tickets ORDER BY created_at DESC LIMIT ?
                    )
                    """,
                    (self.max_rows,),
                )

    def list_tickets(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM tickets ORDER BY created_at DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_ticket(row) for row in rows]

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()

        return self._row_to_ticket(row) if row else None

    def add_message(self, ticket_id: str, sender: str, role: str, message: str, created_at: str) -> dict[str, Any]:
        with self._lock:
            with self._connect() as conn:
                ticket_exists = conn.execute("SELECT 1 FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
                if not ticket_exists:
                    raise ValueError(f"Ticket not found: {ticket_id}")

                cursor = conn.execute(
                    """
                    INSERT INTO ticket_messages (ticket_id, sender, role, message, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (ticket_id, sender, role, message, created_at),
                )
                message_id = cursor.lastrowid

                row = conn.execute(
                    """
                    SELECT id, ticket_id, sender, role, message, created_at
                    FROM ticket_messages
                    WHERE id = ?
                    """,
                    (message_id,),
                ).fetchone()

        return self._row_to_message(row)

    def list_messages(self, ticket_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, ticket_id, sender, role, message, created_at
                FROM ticket_messages
                WHERE ticket_id = ?
                ORDER BY id ASC
                """,
                (ticket_id,),
            ).fetchall()

        return [self._row_to_message(row) for row in rows]

    def _row_to_ticket(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "channel": row["channel"],
            "language": row["language"],
            "category": row["category"],
            "status": row["status"],
            "priority": row["priority"],
            "cleaned_text": row["cleaned_text"],
            "confidence": float(row["confidence"]),
            "processing_timeline": json.loads(row["processing_timeline_json"]),
            "total_processing_ms": int(row["total_processing_ms"]),
            "created_at": datetime.fromisoformat(row["created_at"]),
        }

    def _row_to_message(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "ticket_id": row["ticket_id"],
            "sender": row["sender"],
            "role": row["role"],
            "message": row["message"],
            "created_at": row["created_at"],
        }

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
