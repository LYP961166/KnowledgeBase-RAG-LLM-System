"""向量库的构建与检索入口。

集合在这里统一创建，保证「写入端」与「检索端」使用同一套距离度量；
检索对外只暴露 search()，内部走 retrieval.HybridRetriever。
"""

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

import config_data as config
from retrieval import HybridRetriever


def build_chroma(embedding_function=None) -> Chroma:
    """创建或打开集合。距离度量只在集合首次创建时生效。"""
    return Chroma(
        collection_name=config.collection_name,
        embedding_function=embedding_function,
        persist_directory=config.persist_directory,
        client_settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=config.persist_directory,
        ),
        collection_metadata={"hnsw:space": config.distance_metric},
    )


class VectorStoreService:
    def __init__(self) -> None:
        self.vector_store = build_chroma(None)
        self._retriever: HybridRetriever | None = None

    def count(self) -> int:
        return len(self.vector_store.get(include=[]).get("ids") or [])

    def all_documents(self):
        """整库读取，用于构建 BM25 索引。"""
        rows = self.vector_store.get(include=["documents", "metadatas"])
        return (
            rows.get("ids") or [],
            rows.get("documents") or [],
            rows.get("metadatas") or [],
        )

    def vector_search_raw(self, query: str, k: int):
        """纯向量召回，返回 [(Document, relevance_score)]。"""
        search_store = build_chroma(
            DashScopeEmbeddings(model=config.embedding_model_name)
        )
        return search_store.similarity_search_with_relevance_scores(query, k=k)

    def search(self, query: str):
        """混合检索入口，返回 [(Document, score)]。"""
        if self.count() == 0:
            return []
        if self._retriever is None:
            self._retriever = HybridRetriever(self)
        return self._retriever.search(query)
