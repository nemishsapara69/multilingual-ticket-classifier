import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any


class AuthStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )

    def ensure_user(self, username: str, password: str, role: str) -> None:
        existing = self.get_user(username)
        if existing:
            return

        salt = secrets.token_hex(16)
        password_hash = _hash_password(password, salt)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, role, disabled) VALUES (?, ?, ?, ?, 0)",
                (username, password_hash, salt, role),
            )

    def get_user(self, username: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT username, password_hash, salt, role, disabled FROM users WHERE username = ?",
                (username,),
            ).fetchone()

        if not row:
            return None

        return {
            "username": row["username"],
            "password_hash": row["password_hash"],
            "salt": row["salt"],
            "role": row["role"],
            "disabled": bool(row["disabled"]),
        }

    def verify_credentials(self, username: str, password: str) -> dict[str, Any] | None:
        user = self.get_user(username)
        if not user or user["disabled"]:
            return None

        computed = _hash_password(password, user["salt"])
        if not secrets.compare_digest(computed, user["password_hash"]):
            return None

        return {"username": user["username"], "role": user["role"]}

    def create_session(self, username: str, role: str, ttl_hours: int = 24) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat()

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (token, username, role, expires_at) VALUES (?, ?, ?, ?)",
                (token, username, role, expires_at),
            )

        return token

    def get_session(self, token: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT token, username, role, expires_at FROM sessions WHERE token = ?",
                (token,),
            ).fetchone()

            if not row:
                return None

            expires_at = datetime.fromisoformat(row["expires_at"])
            if expires_at < datetime.now(timezone.utc):
                conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
                return None

        return {
            "token": row["token"],
            "username": row["username"],
            "role": row["role"],
            "expires_at": row["expires_at"],
        }

    def revoke_session(self, token: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def _hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200_000,
    )
    return digest.hex()
