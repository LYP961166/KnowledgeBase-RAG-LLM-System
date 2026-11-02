"""LangChain 消息历史，底层落在 SQLite 会话库。

作用：让 RunnableWithMessageHistory 读到的历史与界面上看到的会话是同一份数据，
同时把「只取最近 N 条」的窗口控制放在读取侧，避免提示词随对话轮次无限增长。
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

import config_data as config
from storage import ConversationStore

_store: ConversationStore | None = None


def get_store() -> ConversationStore:
    """进程内共享一个存储实例，避免每次取历史都重建表结构。"""
    global _store
    if _store is None:
        _store = ConversationStore(config.db_path)
    return _store


class SqliteChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, conversation_id: str, max_messages: int | None = None) -> None:
        self.conversation_id = conversation_id
        self.max_messages = max_messages or config.max_history_messages

    @property
    def messages(self) -> list[BaseMessage]:
        rows = get_store().get_messages(self.conversation_id, limit=self.max_messages)
        history: list[BaseMessage] = []
        for row in rows:
            if row["role"] == "user":
                history.append(HumanMessage(content=row["content"]))
            else:
                history.append(AIMessage(content=row["content"]))
        return history

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        store = get_store()
        for message in messages:
            role = "user" if message.type == "human" else "assistant"
            store.add_message(self.conversation_id, role, message.content)

    def clear(self) -> None:
        get_store().clear_messages(self.conversation_id)


def get_history(conversation_id: str) -> SqliteChatMessageHistory:
    return SqliteChatMessageHistory(conversation_id)
