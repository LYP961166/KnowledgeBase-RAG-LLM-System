import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_base import KnowledgeBaseService


class FakeChroma:
    def __init__(self):
        self.rows = {}
        self.batch_sizes = []

    def get(self, ids=None, where=None, limit=None, include=None, **kwargs):
        rows = list(self.rows.items())
        if ids is not None:
            wanted = set(ids)
            rows = [(row_id, row) for row_id, row in rows if row_id in wanted]
        if where:
            rows = [
                (row_id, row)
                for row_id, row in rows
                if all(row["metadata"].get(key) == value for key, value in where.items())
            ]
        if limit is not None:
            rows = rows[:limit]
        return {
            "ids": [row_id for row_id, _ in rows],
            "metadatas": [row["metadata"] for _, row in rows],
        }

    def add_texts(self, texts, metadatas, ids):
        self.batch_sizes.append(len(texts))
        for row_id, text, metadata in zip(ids, texts, metadatas):
            self.rows[row_id] = {"text": text, "metadata": metadata}
        return ids

    def delete(self, ids=None, **kwargs):
        for row_id in ids or []:
            self.rows.pop(row_id, None)


class KnowledgeBaseServiceTests(unittest.TestCase):
    def setUp(self):
        self.db = FakeChroma()
        self.service = KnowledgeBaseService(chroma=self.db)

    def test_upload_list_duplicate_delete_and_clear(self):
        self.assertEqual(
            self.service.get_status(), {"document_count": 0, "chunk_count": 0}
        )

        first = self.service.upload_by_str("粤剧是岭南传统戏曲。" * 80, "粤剧.txt")
        self.assertEqual(first["status"], "success")
        self.assertGreater(first["chunk_count"], 1)
        self.assertEqual(self.service.get_status()["document_count"], 1)

        rows = self.service.list_documents()
        self.assertEqual(rows[0]["source"], "粤剧.txt")
        self.assertTrue(rows[0]["file_hash"])
        self.assertEqual(rows[0]["chunk_count"], first["chunk_count"])

        duplicate = self.service.upload_by_str("粤剧是岭南传统戏曲。" * 80, "副本.txt")
        self.assertEqual(duplicate["status"], "repeat")

        second = self.service.upload_by_str("广东醒狮资料", "醒狮.txt")
        self.assertEqual(second["status"], "success")
        self.assertEqual(self.service.get_status()["document_count"], 2)
        self.assertTrue(self.service.delete_document(first["file_hash"]))
        self.assertEqual(self.service.list_documents()[0]["source"], "醒狮.txt")
        self.assertEqual(self.service.clear(), 1)
        self.assertEqual(self.service.get_status()["chunk_count"], 0)

    def test_rejects_blank_document(self):
        result = self.service.upload_by_str(" \n\n ", "空文件.txt")
        self.assertEqual(result["status"], "error")

    def test_embedding_batches_never_exceed_dashscope_limit(self):
        text = "\n\n".join(
            f"第{index}段：" + ("岭南非遗文化景点资料。" * 80)
            for index in range(12)
        )
        result = self.service.upload_by_str(text, "大文档.txt")
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["chunk_count"], 10)
        self.assertTrue(self.db.batch_sizes)
        self.assertLessEqual(max(self.db.batch_sizes), 10)


if __name__ == "__main__":
    unittest.main()
