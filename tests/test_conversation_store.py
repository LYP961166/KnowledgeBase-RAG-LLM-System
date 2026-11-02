"""会话存储与 LangChain 历史层的单元测试。"""

import sys
import tempfile
import unittest
from pathlib import Path

test_dir = Path(__file__).resolve().parent
project_root = test_dir if (test_dir / "storage.py").exists() else test_dir.parent
sys.path.insert(0, str(project_root))

import config_data as config
import sqlite_history_store
from sqlite_history_store import SqliteChatMessageHistory, get_history
from storage import ConversationStore


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "test.db")
        self.store = ConversationStore(self.db_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_create_and_list(self):
        first = self.store.create_conversation()
        second = self.store.create_conversation()
        items = self.store.list_conversations()
        self.assertEqual(len(items), 2)
        self.assertEqual({item["id"] for item in items}, {first, second})
        self.assertEqual(items[0]["message_count"], 0)

    def test_default_title_replaced_by_first_question(self):
        conversation_id = self.store.create_conversation()
        self.assertEqual(
            self.store.get_conversation(conversation_id)["title"],
            config.default_conversation_title,
        )
        self.store.add_message(conversation_id, "user", "陈家祠有哪些非遗项目？")
        self.assertEqual(self.store.get_conversation(conversation_id)["title"], "陈家祠有哪些非遗项目？")

    def test_manual_title_not_overwritten(self):
        conversation_id = self.store.create_conversation()
        self.store.rename_conversation(conversation_id, "广州非遗专题")
        self.store.add_message(conversation_id, "user", "醒狮和舞狮有什么区别")
        self.assertEqual(self.store.get_conversation(conversation_id)["title"], "广州非遗专题")

    def test_add_exchange_writes_two_messages(self):
        conversation_id = self.store.create_conversation()
        self.store.add_exchange(conversation_id, "你好", "你好，我是岭南非遗知识助手。")
        messages = self.store.get_messages(conversation_id)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertEqual(messages[1]["content"], "你好，我是岭南非遗知识助手。")

    def test_get_messages_limit_returns_recent_in_order(self):
        conversation_id = self.store.create_conversation()
        for index in range(6):
            self.store.add_message(conversation_id, "user", f"问题{index}")
        recent = self.store.get_messages(conversation_id, limit=3)
        self.assertEqual([m["content"] for m in recent], ["问题3", "问题4", "问题5"])

    def test_attach_sources_to_last_assistant_message(self):
        conversation_id = self.store.create_conversation()
        self.store.add_exchange(conversation_id, "问题", "回答")
        payload = [{"source": "岭南非遗.txt", "score": 0.82, "preview": "片段"}]
        self.assertTrue(
            self.store.set_last_assistant_meta(
                conversation_id, sources=payload, search_query="改写后的问题", intent="knowledge"
            )
        )
        assistant = self.store.get_messages(conversation_id)[-1]
        self.assertEqual(assistant["sources"][0]["source"], "岭南非遗.txt")
        self.assertEqual(assistant["search_query"], "改写后的问题")
        self.assertEqual(assistant["intent"], "knowledge")

    def test_feedback_persisted(self):
        conversation_id = self.store.create_conversation()
        self.store.add_exchange(conversation_id, "问题", "回答")
        assistant = self.store.get_messages(conversation_id)[-1]
        self.store.set_feedback(assistant["id"], -1)
        self.assertEqual(self.store.get_messages(conversation_id)[-1]["feedback"], -1)

    def test_delete_conversation_removes_messages(self):
        conversation_id = self.store.create_conversation()
        self.store.add_exchange(conversation_id, "问题", "回答")
        self.assertTrue(self.store.delete_conversation(conversation_id))
        self.assertIsNone(self.store.get_conversation(conversation_id))
        self.assertEqual(self.store.get_messages(conversation_id), [])

    def test_survives_reopen(self):
        conversation_id = self.store.create_conversation()
        self.store.add_exchange(conversation_id, "问题", "回答")
        reopened = ConversationStore(self.db_path)
        self.assertEqual(len(reopened.get_messages(conversation_id)), 2)


class SqliteHistoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db = config.db_path
        self._original_window = config.max_history_messages
        config.db_path = str(Path(self._tmp.name) / "history.db")
        sqlite_history_store._store = None
        self.store = sqlite_history_store.get_store()
        self.conversation_id = self.store.create_conversation()

    def tearDown(self):
        config.db_path = self._original_db
        config.max_history_messages = self._original_window
        sqlite_history_store._store = None
        self._tmp.cleanup()

    def test_empty_history(self):
        self.assertEqual(get_history(self.conversation_id).messages, [])

    def test_added_messages_are_read_back_with_roles(self):
        from langchain_core.messages import AIMessage, HumanMessage

        history = SqliteChatMessageHistory(self.conversation_id)
        history.add_messages([HumanMessage(content="你好"), AIMessage(content="在的")])
        messages = get_history(self.conversation_id).messages
        self.assertEqual([m.type for m in messages], ["human", "ai"])
        self.assertEqual(messages[0].content, "你好")

    def test_window_limits_prompt_history(self):
        from langchain_core.messages import HumanMessage

        config.max_history_messages = 3
        history = SqliteChatMessageHistory(self.conversation_id)
        for index in range(5):
            history.add_messages([HumanMessage(content=f"第{index}条")])
        messages = get_history(self.conversation_id).messages
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[-1].content, "第4条")

    def test_clear_removes_conversation_messages(self):
        from langchain_core.messages import HumanMessage

        history = SqliteChatMessageHistory(self.conversation_id)
        history.add_messages([HumanMessage(content="待清空")])
        history.clear()
        self.assertEqual(get_history(self.conversation_id).messages, [])


if __name__ == "__main__":
    unittest.main()
