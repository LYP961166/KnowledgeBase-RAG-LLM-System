"""混合检索测试：分词、BM25、RRF 融合与重排降级。"""

import sys
import unittest
from pathlib import Path

test_dir = Path(__file__).resolve().parent
project_root = test_dir if (test_dir / "retrieval.py").exists() else test_dir.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document

import config_data as config
import retrieval
from retrieval import (
    BM25Index,
    HybridRetriever,
    document_key,
    reciprocal_rank_fusion,
    tokenize,
)


class TokenizeTests(unittest.TestCase):
    def test_latin_words_kept_whole(self):
        self.assertIn("rag", tokenize("RAG 检索"))

    def test_chinese_produces_single_chars_and_bigrams(self):
        tokens = tokenize("香云纱")
        self.assertIn("香", tokens)
        self.assertIn("香云", tokens)
        self.assertIn("云纱", tokens)


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "陈家祠展示广绣、广彩与石湾陶塑",
            "永庆坊有粤剧艺术博物馆",
            "广东省非遗馆综合展示各类非遗项目",
        ]
        self.index = BM25Index([tokenize(text) for text in self.corpus])

    def test_ranks_matching_document_first(self):
        ranked = self.index.search(tokenize("粤剧艺术博物馆"), top_k=3)
        self.assertTrue(ranked)
        self.assertEqual(ranked[0][0], 1)

    def test_no_match_returns_empty(self):
        self.assertEqual(self.index.search(tokenize("量子计算"), top_k=3), [])

    def test_empty_corpus_is_safe(self):
        empty = BM25Index([])
        self.assertEqual(empty.search(tokenize("任意"), top_k=3), [])


class FusionTests(unittest.TestCase):
    def test_fuses_and_dedupes(self):
        vector_rank = [("a", "docA"), ("b", "docB")]
        keyword_rank = [("b", "docB"), ("c", "docC")]
        fused = reciprocal_rank_fusion([vector_rank, keyword_rank], k=60)
        keys = [key for key, _, _ in fused]
        self.assertEqual(len(keys), 3)
        self.assertEqual(len(set(keys)), 3)
        # b 在两路都出现，融合后排第一
        self.assertEqual(keys[0], "b")

    def test_empty_input(self):
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])


class DocumentKeyTests(unittest.TestCase):
    def test_uses_chunk_identity_when_available(self):
        doc = Document(page_content="x", metadata={"file_hash": "h1", "chunk_index": 3})
        self.assertEqual(document_key(doc), "h1:3")

    def test_falls_back_to_content_hash(self):
        doc = Document(page_content="相同内容", metadata={})
        self.assertEqual(document_key(doc), document_key(Document(page_content="相同内容", metadata={})))


class FakeVectorService:
    def __init__(self, docs):
        self._rows = docs

    def count(self):
        return len(self._rows)

    def all_documents(self):
        ids = [row[0] for row in self._rows]
        texts = [row[1] for row in self._rows]
        metas = [row[2] for row in self._rows]
        return ids, texts, metas

    def vector_search_raw(self, query, k):
        return [(Document(page_content=row[1], metadata=row[2]), 0.8) for row in self._rows[:k]]


class FakeReranker:
    def __init__(self, scores=None):
        self.scores = scores
        self.called = 0

    def available(self):
        return True

    def rerank(self, query, documents, top_n):
        self.called += 1
        if self.scores is None:
            return None
        return [(doc, self.scores[i]) for i, doc in enumerate(documents[:top_n])]


def make_service():
    rows = [
        ("h1:0", "陈家祠 展示 广绣 广彩 石湾陶塑", {"file_hash": "h1", "chunk_index": 0}),
        ("h2:0", "永庆坊 粤剧 艺术博物馆 戏曲", {"file_hash": "h2", "chunk_index": 0}),
        ("h3:0", "香云纱 染整 技艺 顺德", {"file_hash": "h3", "chunk_index": 0}),
    ]
    return FakeVectorService(rows)


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self):
        self._top_k = config.retrieval_top_k
        self._min = config.rerank_min_score

    def tearDown(self):
        config.retrieval_top_k = self._top_k
        config.rerank_min_score = self._min

    def test_empty_query_returns_nothing(self):
        retriever = HybridRetriever(make_service(), FakeReranker(scores=[0.9]))
        self.assertEqual(retriever.search("   "), [])

    def test_no_documents_returns_nothing(self):
        retriever = HybridRetriever(FakeVectorService([]), FakeReranker(scores=[0.9]))
        self.assertEqual(retriever.search("粤剧"), [])

    def test_falls_back_when_rerank_fails(self):
        retriever = HybridRetriever(make_service(), FakeReranker(scores=None))
        results = retriever.search("香云纱")
        self.assertTrue(results)
        scores = [score for _, score in results]
        self.assertEqual(max(scores), 1.0)
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))

    def test_uses_reranked_order_and_scores(self):
        retriever = HybridRetriever(make_service(), FakeReranker(scores=[0.62, 0.41, 0.09]))
        results = retriever.search("粤剧")
        self.assertTrue(results)
        self.assertAlmostEqual(results[0][1], 0.62)

    def test_filters_below_min_score(self):
        config.rerank_min_score = 0.5
        retriever = HybridRetriever(make_service(), FakeReranker(scores=[0.62, 0.41, 0.09]))
        results = retriever.search("粤剧")
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0][1], 0.62)

    def test_rerank_not_called_when_disabled(self):
        original = config.rerank_enabled
        config.rerank_enabled = False
        try:
            reranker = FakeReranker(scores=[0.9])
            retriever = HybridRetriever(make_service(), reranker)
            retriever.search("粤剧")
            self.assertEqual(reranker.called, 0)
        finally:
            config.rerank_enabled = original

    def test_index_rebuilt_when_corpus_changes(self):
        service = make_service()
        retriever = HybridRetriever(service, FakeReranker(scores=None))
        retriever.search("粤剧")
        first_size = len(retriever._documents)
        service._rows = service._rows + [
            ("h4:0", "广彩 瓷器 广州", {"file_hash": "h4", "chunk_index": 0})
        ]
        retriever.search("广彩")
        self.assertEqual(len(retriever._documents), first_size + 1)


if __name__ == "__main__":
    unittest.main()
