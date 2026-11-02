"""SQLite 会话与消息存储。

会话管理的唯一数据源：会话列表、消息记录、引用来源、反馈都落在这两张表里。
界面层与 LangChain 历史层都从这里读写，避免同一条消息被写入两次。
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

import config_data as config

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    sources         TEXT NOT NULL DEFAULT '[]',
    search_query    TEXT NOT NULL DEFAULT '',
    intent          TEXT NOT NULL DEFAULT '',
    feedback        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, id);
"""


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ConversationStore:
    """会话与消息的增删改查。每次操作使用独立连接，可安全用于多线程环境。"""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or config.db_path
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------- 会话 ----------------

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM messages m
                        WHERE m.conversation_id = c.id) AS message_count
                FROM conversations c
                ORDER BY c.updated_at DESC, c.created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        if not conversation_id:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def create_conversation(self, title: str | None = None) -> str:
        conversation_id = uuid.uuid4().hex
        stamp = now_text()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, title or config.default_conversation_title, stamp, stamp),
            )
        return conversation_id

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        title = (title or "").strip()
        if not title:
            return False
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title[:60], now_text(), conversation_id),
            )
        return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
        return cursor.rowcount > 0

    def touch(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now_text(), conversation_id),
            )

    # ---------------- 消息 ----------------

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
        search_query: str = "",
        intent: str = "",
    ) -> int:
        stamp = now_text()
        payload = json.dumps(sources or [], ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages
                    (conversation_id, role, content, sources, search_query, intent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, payload, search_query, intent, stamp),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (stamp, conversation_id),
            )
            if role == "user":
                self._auto_title(conn, conversation_id, content)
        return int(cursor.lastrowid)

    @staticmethod
    def _auto_title(conn: sqlite3.Connection, conversation_id: str, first_text: str) -> None:
        """第一条用户消息决定会话标题，用户手动改名后不再覆盖。"""
        row = conn.execute(
            "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if not row or row["title"] != config.default_conversation_title:
            return
        title = " ".join((first_text or "").split())[: config.conversation_title_chars]
        if title:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id),
            )

    def add_exchange(
        self,
        conversation_id: str,
        user_text: str,
        assistant_text: str,
        sources: list[dict[str, Any]] | None = None,
        search_query: str = "",
        intent: str = "",
    ) -> None:
        """一次写入一问一答，用于不经过生成链的路由分支。"""
        self.add_message(conversation_id, "user", user_text)
        self.add_message(
            conversation_id,
            "assistant",
            assistant_text,
            sources=sources,
            search_query=search_query,
            intent=intent,
        )

    def get_messages(
        self, conversation_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        if not conversation_id:
            return []
        with self._connect() as conn:
            if limit:
                rows = conn.execute(
                    "SELECT * FROM (SELECT * FROM messages WHERE conversation_id = ? "
                    "ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                    (conversation_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                    (conversation_id,),
                ).fetchall()
        messages = []
        for row in rows:
            item = dict(row)
            try:
                item["sources"] = json.loads(item["sources"] or "[]")
            except json.JSONDecodeError:
                item["sources"] = []
            messages.append(item)
        return messages

    def set_last_assistant_meta(
        self,
        conversation_id: str,
        sources: list[dict[str, Any]] | None = None,
        search_query: str = "",
        intent: str = "",
    ) -> bool:
        """把引用来源等信息补写到最近一条助手消息上（生成链只负责落正文）。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM messages WHERE conversation_id = ? AND role = 'assistant' "
                "ORDER BY id DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
            if not row:
                return False
            conn.execute(
                "UPDATE messages SET sources = ?, search_query = ?, intent = ? WHERE id = ?",
                (
                    json.dumps(sources or [], ensure_ascii=False),
                    search_query,
                    intent,
                    row["id"],
                ),
            )
        return True

    def set_feedback(self, message_id: int, value: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE messages SET feedback = ? WHERE id = ?", (int(value), int(message_id))
            )

    def clear_messages(self, conversation_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
        return cursor.rowcount
