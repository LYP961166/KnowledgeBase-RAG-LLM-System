"""混合检索：向量召回 + BM25 关键词召回 + RRF 融合 + 交叉编码器重排。

为什么需要混合：
- 向量召回擅长语义相近，但对"香云纱""广彩"这类低频专有名词容易失焦；
- BM25 依赖字面命中，恰好补上这一块，但对同义改写无能为力。
两者融合后再重排，才能兼顾语义与精确匹配。
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document

import config_data as config

# 中文不分词，用「单字 + 相邻二字」组合近似切词：
# 单字保证召回，二字组合恢复"香云纱""陈家祠"这类词的区分度，且不依赖 jieba 词典。
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    latin = LATIN_RE.findall(text)
    cjk_chars = CJK_RE.findall(text)
    bigrams = [a + b for a, b in zip(cjk_chars, cjk_chars[1:])]
    return latin + cjk_chars + bigrams


class BM25Index:
    """标准 Okapi BM25，语料规模在万级以内时性能足够。"""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [list(tokens) for tokens in corpus]
        self.size = len(self.corpus)
        if not self.size:
            self.avgdl = 0.0
            self.term_freqs: list[Counter] = []
            self.doc_lens: list[int] = []
            self.idf: dict[str, float] = {}
            return

        self.doc_lens = [len(tokens) for tokens in self.corpus]
        self.avgdl = sum(self.doc_lens) / self.size
        self.term_freqs = [Counter(tokens) for tokens in self.corpus]

        doc_freq: Counter = Counter()
        for tokens in self.corpus:
            doc_freq.update(set(tokens))
        self.idf = {
            term: math.log(1 + (self.size - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def search(self, query_tokens: Iterable[str], top_k: int = 10) -> list[tuple[int, float]]:
        if not self.size:
            return []
        scores: list[tuple[int, float]] = []
        for index in range(self.size):
            freq = self.term_freqs[index]
            doc_len = self.doc_lens[index]
            total = 0.0
            for term in query_tokens:
                tf = freq.get(term)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                total += idf * tf * (self.k1 + 1) / denominator
            if total > 0:
                scores.append((index, total))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[str, Any]]], k: int | None = None
) -> list[tuple[str, Any, float]]:
    """倒数排名融合：只看名次不看分值，避免两路分数量纲不同无法比较。"""
    k = k or config.rrf_k
    fused: defaultdict[str, float] = defaultdict(float)
    payload: dict[str, Any] = {}
    for ranked in ranked_lists:
        for rank, (key, item) in enumerate(ranked, start=1):
            fused[key] += 1.0 / (k + rank)
            payload.setdefault(key, item)
    ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
    return [(key, payload[key], score) for key, score in ordered]


def document_key(document: Document) -> str:
    metadata = document.metadata or {}
    if metadata.get("file_hash") and metadata.get("chunk_index") is not None:
        return f"{metadata['file_hash']}:{metadata['chunk_index']}"
    return hashlib.md5(document.page_content.encode("utf-8")).hexdigest()


class DashScopeReranker:
    """交叉编码器重排。调用失败时返回 None，由调用方退回融合排序。"""

    def available(self) -> bool:
        if not getattr(config, "rerank_enabled", False):
            return False
        return bool(os.getenv("DASHSCOPE_API_KEY", "").strip())

    def rerank(
        self, query: str, documents: Sequence[Document], top_n: int
    ) -> list[tuple[Document, float]] | None:
        if not documents or not self.available():
            return None
        try:
            from dashscope import TextReRank

            response = TextReRank.call(
                model=config.rerank_model_name,
                query=query,
                documents=[document.page_content for document in documents],
                top_n=min(top_n, len(documents)),
                return_documents=False,
            )
            if getattr(response, "status_code", 200) != 200:
                raise RuntimeError(getattr(response, "message", "unknown error"))
            results = response.output.results or []
            ranked: list[tuple[Document, float]] = []
            for item in results:
                index = item.get("index")
                score = item.get("relevance_score", item.get("score", 0.0))
                if index is None or not (0 <= index < len(documents)):
                    continue
                ranked.append((documents[index], float(score)))
            return ranked or None
        except Exception as exc:  # 网络、鉴权、额度问题都不应中断问答
            print(f"Rerank failed: {type(exc).__name__}: {exc}")
            return None


class HybridRetriever:
    """对外只暴露 search()，返回 [(Document, score)]，与原有调用方兼容。"""

    def __init__(self, vector_service: Any, reranker: DashScopeReranker | None = None):
        self.vector_service = vector_service
        self.reranker = reranker or DashScopeReranker()
        self._bm25: BM25Index | None = None
        self._documents: list[Document] = []
        self._signature: int | None = None

    # ---------------- 索引 ----------------
    def _build_index(self) -> None:
        """从向量库整库读出片段重建 BM25 索引；文档增删后靠片段数变化触发重建。"""
        _, documents, metadatas = self.vector_service.all_documents()
        self._documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(documents, metadatas)
        ]
        self._bm25 = BM25Index([tokenize(text) for text in documents])
        self._signature = len(documents)

    def _ensure(self) -> None:
        if self._bm25 is None or self._signature != self.vector_service.count():
            self._build_index()

    # ---------------- 两路召回 ----------------
    def _vector_candidates(self, query: str) -> list[tuple[str, Document]]:
        hits = self.vector_service.vector_search_raw(query, config.vector_candidates)
        return [(document_key(document), document) for document, _ in hits]

    def _keyword_candidates(self, query: str) -> list[tuple[str, Document]]:
        if self._bm25 is None or not self._documents:
            return []
        ranked = self._bm25.search(tokenize(query), config.keyword_candidates)
        return [(document_key(self._documents[index]), self._documents[index])
                for index, _ in ranked]

    # ---------------- 对外 ----------------
    def search(self, query: str, top_k: int | None = None) -> list[tuple[Document, float]]:
        top_k = top_k or config.retrieval_top_k
        if not (query or "").strip():
            return []
        self._ensure()
        if not self._documents:
            return []

        fused = reciprocal_rank_fusion(
            [self._vector_candidates(query), self._keyword_candidates(query)]
        )
        if not fused:
            return []

        candidates = [(document, score) for _, document, score in fused]
        pool = candidates[: config.rerank_candidates]

        reranked = None
        if getattr(config, "rerank_enabled", False) and self.reranker.available():
            reranked = self.reranker.rerank(
                query, [document for document, _ in pool], top_k
            )
        if reranked:
            kept = [(document, score) for document, score in reranked
                    if score >= config.rerank_min_score]
            return (kept or reranked)[:top_k]

        # 重排不可用时，用融合分归一化后当作展示分
        best = pool[0][1] if pool else 1.0
        return [(document, (score / best if best else 0.0)) for document, score in pool[:top_k]]
