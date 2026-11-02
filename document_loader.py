"""把上传的文件字节流解析为纯文本。

只负责「文件 → 文本」这一步，切分、去重、写库仍由 knowledge_base 负责，
这样新增一种格式时只需要在这里加一个分支。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

# 界面上展示的格式清单，顺序即提示顺序
SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF",
    ".docx": "Word",
    ".txt": "TXT",
}

UPLOAD_TYPES = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]


class UnsupportedDocument(Exception):
    """格式不支持或无法解析，消息可以直接展示给用户。"""


@dataclass
class ParsedDocument:
    text: str
    file_type: str
    pages: int = 0


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnsupportedDocument("文本文件编码无法识别，请另存为 UTF-8 后重新上传。")


def _read_pdf(data: bytes) -> ParsedDocument:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = len(reader.pages)
        texts = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:  # pypdf 对损坏文件抛出的异常类型不固定
        raise UnsupportedDocument(f"PDF 解析失败：{type(exc).__name__}") from exc

    body = "\n\n".join(text for text in texts if text)
    if not body.strip():
        raise UnsupportedDocument(
            "这份 PDF 没有可提取的文字，可能是扫描件或纯图片，需要先做 OCR。"
        )
    return ParsedDocument(text=body, file_type="PDF", pages=pages)


def _read_docx(data: bytes) -> ParsedDocument:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise UnsupportedDocument(f"Word 解析失败：{type(exc).__name__}") from exc

    blocks = [paragraph.text.strip() for paragraph in document.paragraphs]
    # 表格里的非遗项目名录、级别、批次往往是结构化数据，一并取出
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            line = " | ".join(cell for cell in cells if cell)
            if line:
                blocks.append(line)

    body = "\n".join(block for block in blocks if block)
    if not body.strip():
        raise UnsupportedDocument("这份 Word 文档没有正文文字，请检查内容后重试。")
    return ParsedDocument(text=body, file_type="Word")


def extract_text(filename: str, data: bytes) -> ParsedDocument:
    """按扩展名选择解析器，返回归一化后的正文。"""
    if not data:
        raise UnsupportedDocument("文件内容为空。")

    extension = Path(filename).suffix.lower()
    if extension == ".doc":
        raise UnsupportedDocument(
            ".doc 是旧版二进制格式，请在 Word 中另存为 .docx 后再上传。"
        )
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = "、".join(SUPPORTED_EXTENSIONS.values())
        raise UnsupportedDocument(f"暂不支持该格式，目前可上传：{allowed}。")

    if extension == ".pdf":
        parsed = _read_pdf(data)
    elif extension == ".docx":
        parsed = _read_docx(data)
    else:
        parsed = ParsedDocument(text=_decode_text(data), file_type="TXT")

    parsed.text = _normalize(parsed.text)
    if not parsed.text:
        raise UnsupportedDocument("文件里没有可录入的文字内容。")
    return parsed
