"""结构化文本切分。

优先按文档自身的层级（章 / 条目）切分，让一个知识片段对应一个完整语义单元；
识别不到结构时自动退回递归切分，保证任意纯文本也能用。

设计取舍：
- 每个片段开头补一行「地区·标题」前缀，使向量同时编码"这段属于谁"，缓解指代缺失；
- 切分单位本身已自包含，因此结构化路径不设重叠，避免同一段内容被重复计费与重复召回。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config

# 一级：地区 / 章节
SECTION_PATTERNS = (
    re.compile(r"^[一二三四五六七八九十百]+、\s*(.+)$"),
    re.compile(r"^第[一二三四五六七八九十百]+[章节部分篇]\s*(.*)$"),
    re.compile(r"^#{1,3}\s+(.+)$"),
)

# 二级：条目 / 小节
ENTRY_PATTERNS = (
    re.compile(r"^\d{2,}｜\s*(.+)$"),
    re.compile(r"^【([^】]{2,40})】$"),
    re.compile(r"^\d+\.\d+\s+(.+)$"),
)

SEPARATOR_LINE = re.compile(r"^[=\-_*~]{6,}$")

MIN_TITLE_COVERAGE = 0.5


@dataclass
class TextChunk:
    text: str
    section: str = ""
    title: str = ""

    @property
    def header(self) -> str:
        if self.section and self.title:
            return f"【{self.section}·{self.title}】"
        if self.title:
            return f"【{self.title}】"
        return ""


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _match_section(line: str) -> str | None:
    for pattern in SECTION_PATTERNS:
        found = pattern.match(line)
        if found:
            return (found.group(1) or line).strip()
    return None


def _match_entry(line: str) -> str | None:
    for pattern in ENTRY_PATTERNS:
        found = pattern.match(line)
        if found:
            return (found.group(1) or line).strip()
    return None


def _parse_blocks(text: str) -> list[tuple[str, str, list[str]]]:
    """把文本拆成 (所属章节, 条目标题, 正文行) 三元组。"""
    blocks: list[tuple[str, str, list[str]]] = []
    section = ""
    title = ""
    body: list[str] = []

    def flush() -> None:
        nonlocal body, title
        if title or body:
            blocks.append((section, title, body))
        body = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or SEPARATOR_LINE.match(line):
            continue

        section_name = _match_section(line)
        if section_name:
            flush()
            section, title = section_name, ""
            continue

        entry_name = _match_entry(line)
        if entry_name:
            flush()
            title = entry_name
            continue

        body.append(line)
    flush()
    return blocks


def _build_header(section: str, title: str) -> str:
    if section and title:
        return f"【{section}·{title}】"
    if title:
        return f"【{title}】"
    if section:
        return f"【{section}】"
    return ""


def _split_long_block(
    header: str, body: list[str], chunk_size: int, section: str, title: str
) -> list[TextChunk]:
    """超长条目按行二次切分，每片都带上标题前缀，避免后半段失去归属。"""
    pieces: list[TextChunk] = []
    buffer: list[str] = []
    for line in body:
        candidate = "\n".join(buffer + [line])
        if buffer and len(header) + 1 + len(candidate) > chunk_size:
            pieces.append(
                TextChunk(text=f"{header}\n" + "\n".join(buffer), section=section, title=title)
            )
            buffer = []
        # 单行本身超长时按句号硬切
        if len(line) > chunk_size:
            sentences = re.split(r"(?<=[。！？；])", line)
            for sentence in sentences:
                if not sentence:
                    continue
                buffer.append(sentence)
                if len("\n".join(buffer)) >= chunk_size:
                    pieces.append(
                        TextChunk(
                            text=f"{header}\n" + "\n".join(buffer),
                            section=section,
                            title=title,
                        )
                    )
                    buffer = []
            continue
        buffer.append(line)

    if buffer:
        pieces.append(
            TextChunk(text=f"{header}\n" + "\n".join(buffer), section=section, title=title)
        )
    return pieces


def _structured_split(text: str, chunk_size: int) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for section, title, body in _parse_blocks(text):
        header = _build_header(section, title)
        payload = "\n".join(body)
        if len(header) + 1 + len(payload) <= chunk_size:
            # 只要条目本身有正文就保留，篇幅短的条目同样可能是有效知识
            if payload.strip():
                content = f"{header}\n{payload}".strip() if header else payload.strip()
                chunks.append(TextChunk(text=content, section=section, title=title))
            continue
        chunks.extend(_split_long_block(header, body, chunk_size, section, title))
    return chunks


def _fallback_split(text: str, chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=config.separators,
        length_function=len,
    )
    return [
        TextChunk(text=piece)
        for piece in splitter.split_text(text)
        if piece.strip()
    ]


def split_document(
    text: str, chunk_size: int | None = None, chunk_overlap: int | None = None
) -> list[TextChunk]:
    """按文档结构切分；结构不足时退回递归切分。"""
    chunk_size = chunk_size or config.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
    normalized = normalize(text)
    if not normalized:
        return []

    if getattr(config, "chunking_mode", "structured") == "fixed":
        return _fallback_split(normalized, chunk_size, chunk_overlap)

    blocks = _parse_blocks(normalized)
    titled_blocks = [block for block in blocks if block[1]]
    # 只有「有标题的条目」覆盖了大部分正文，才认为文档确实具备可利用的层级；
    # 否则说明标题只是零散出现，正文主体仍是无结构段落，交给递归切分更稳。
    covered = sum(len("\n".join(body)) for _, _, body in titled_blocks)
    if not titled_blocks or covered < len(normalized) * MIN_TITLE_COVERAGE:
        return _fallback_split(normalized, chunk_size, chunk_overlap)

    chunks = _structured_split(normalized, chunk_size)
    return chunks or _fallback_split(normalized, chunk_size, chunk_overlap)
