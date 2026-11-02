"""Knowledge-base ingestion and document management."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

from langchain_community.embeddings import DashScopeEmbeddings

import config_data as config
from chunking import split_document
from vector_stores import build_chroma


# DashScope text-embedding-v4 accepts at most 10 input texts per request.
EMBEDDING_BATCH_SIZE = 10


def get_string_md5(input_str: str, encoding: str = "utf-8") -> str:
    """Return a stable content hash used for duplicate detection and deletion."""
    return hashlib.md5(input_str.encode(encoding=encoding)).hexdigest()


class KnowledgeBaseService:
    """Owns ingestion, listing, deletion, and status checks for Chroma."""

    def __init__(self, chroma: Any | None = None) -> None:
        os.makedirs(config.persist_directory, exist_ok=True)
        self._injected_store = chroma is not None
        self.chroma = chroma or build_chroma(None)
        self.build_info_path = os.path.join(config.persist_directory, "build_info.json")

    def get_status(self) -> dict[str, int]:
        documents = self.list_documents()
        return {
            "document_count": len(documents),
            "chunk_count": sum(item["chunk_count"] for item in documents),
        }

    def list_documents(self) -> list[dict[str, Any]]:
        """Aggregate chunk metadata into one row per imported document."""
        result = self.chroma.get(include=["metadatas"])
        grouped: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "source": "未知文档",
                "file_type": "",
                "file_hash": "",
                "create_time": "",
                "chunk_count": 0,
            }
        )
        for metadata in result.get("metadatas") or []:
            metadata = metadata or {}
            file_hash = str(metadata.get("file_hash") or "legacy")
            item = grouped[file_hash]
            item["source"] = str(metadata.get("source") or item["source"])
            item["file_type"] = str(metadata.get("file_type") or item["file_type"])
            item["file_hash"] = "" if file_hash == "legacy" else file_hash
            item["create_time"] = str(
                metadata.get("create_time") or item["create_time"]
            )
            item["chunk_count"] += 1
        return sorted(
            grouped.values(),
            key=lambda item: (item["create_time"], item["source"]),
            reverse=True,
        )

    def upload_by_str(
        self, data: str, filename: str, file_type: str = ""
    ) -> dict[str, Any]:
        """校验、切分并写入一份已经解析成文本的文档。"""
        normalized = data.replace("\r\n", "\n").strip()
        if not normalized:
            return {"status": "error", "message": "文件内容为空，未录入知识库。"}

        file_hash = get_string_md5(normalized)
        duplicate = self.chroma.get(
            where={"file_hash": file_hash}, include=["metadatas"], limit=1
        )
        if duplicate.get("ids"):
            return {"status": "repeat", "message": "相同内容已存在于知识库。"}

        chunks = split_document(normalized)
        if not chunks:
            return {"status": "error", "message": "未切分出有效内容，请检查文档。"}

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadatas = [
            {
                "source": filename,
                "file_type": file_type,
                "file_hash": file_hash,
                "chunk_index": index,
                "chunk_count": len(chunks),
                "section": chunk.section,
                "section_title": chunk.title,
                "create_time": created_at,
            }
            for index, chunk in enumerate(chunks)
        ]
        texts = [chunk.text for chunk in chunks]
        ids = [f"{file_hash}:{index}" for index in range(len(chunks))]
        writer = self.chroma
        if not self._injected_store:
            writer = build_chroma(
                DashScopeEmbeddings(model=config.embedding_model_name)
            )
        written_ids: list[str] = []
        try:
            for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                end = start + EMBEDDING_BATCH_SIZE
                batch_ids = ids[start:end]
                writer.add_texts(
                    texts=texts[start:end],
                    metadatas=metadatas[start:end],
                    ids=batch_ids,
                )
                written_ids.extend(batch_ids)
        except Exception:
            # Keep imports atomic from the user's perspective.  If a later
            # embedding request fails, remove chunks written by earlier batches.
            if written_ids:
                writer.delete(ids=written_ids)
            raise
        if not self._injected_store:
            self._write_build_info()
        return {
            "status": "success",
            "message": f"《{filename}》已录入，共生成 {len(chunks)} 个知识片段。",
            "file_hash": file_hash,
            "chunk_count": len(chunks),
        }

    def delete_document(self, file_hash: str) -> bool:
        if not file_hash:
            return False
        result = self.chroma.get(
            where={"file_hash": file_hash}, include=["metadatas"]
        )
        ids = result.get("ids") or []
        if not ids:
            return False
        self.chroma.delete(ids=ids)
        return True

    def clear(self) -> int:
        """Delete all chunks while keeping the collection available to readers."""
        result = self.chroma.get(include=["metadatas"])
        ids = result.get("ids") or []
        if ids:
            self.chroma.delete(ids=ids)
        return len(ids)

    def current_distance_metric(self) -> str | None:
        metadata = getattr(getattr(self.chroma, "_collection", None), "metadata", None)
        return (metadata or {}).get("hnsw:space")

    def read_build_info(self) -> dict[str, Any] | None:
        """读取建库标记，记录知识库是用哪套参数建起来的。

        不能靠集合元数据判断：chromadb 会把 hnsw:space 自动补写到元数据上，
        但已建好的 HNSW 索引不会因此改变度量，于是出现「元数据写着 cosine、
        实际仍是 L2」的静默错配。用自己写的标记文件才可靠。
        """
        try:
            with open(self.build_info_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _write_build_info(self) -> None:
        payload = {
            "distance_metric": config.distance_metric,
            "chunking_mode": getattr(config, "chunking_mode", "structured"),
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "embedding_model": config.embedding_model_name,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.build_info_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def needs_rebuild(self) -> bool:
        if self._injected_store:
            return False
        info = self.read_build_info()
        has_data = len(self.chroma.get(include=[]).get("ids") or []) > 0
        if info is None:
            return has_data  # 有数据但没有标记：老版本建的库
        return (
            info.get("distance_metric") != config.distance_metric
            or info.get("chunking_mode") != getattr(config, "chunking_mode", "structured")
        )

    def reset_collection(self) -> int:
        """删集合重建，同时清空数据。

        距离度量、切分方式这类建库参数改动后，只有重建集合才会生效；
        clear() 只删片段、保留原集合，改不了这些参数。
        """
        if self._injected_store:
            return self.clear()
        deleted = len(self.chroma.get(include=[]).get("ids") or [])
        # 复用当前客户端的集合句柄，避免同进程内重复创建客户端导致 settings 冲突
        try:
            self.chroma.delete_collection()
        except Exception:
            pass
        self.chroma = build_chroma(None)
        self._write_build_info()
        return deleted
