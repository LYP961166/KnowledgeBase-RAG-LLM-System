"""文档解析层测试：TXT / PDF / Word 三条分支与异常提示。"""

import io
import sys
import unittest
from pathlib import Path

test_dir = Path(__file__).resolve().parent
project_root = test_dir if (test_dir / "document_loader.py").exists() else test_dir.parent
sys.path.insert(0, str(project_root))

import docx
from pypdf import PdfWriter

from document_loader import SUPPORTED_EXTENSIONS, UPLOAD_TYPES, UnsupportedDocument, extract_text


def make_docx(paragraphs, table_rows=None):
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    if table_rows:
        table = document.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r, row in enumerate(table_rows):
            for c, value in enumerate(row):
                table.cell(r, c).text = value
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_blank_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class TxtTests(unittest.TestCase):
    def test_utf8_text(self):
        parsed = extract_text("非遗.txt", "粤剧是岭南代表性非遗项目。".encode("utf-8"))
        self.assertEqual(parsed.file_type, "TXT")
        self.assertIn("粤剧", parsed.text)

    def test_utf8_bom_text(self):
        parsed = extract_text("bom.txt", "\ufeff醒狮属于南狮。".encode("utf-8"))
        self.assertEqual(parsed.text, "醒狮属于南狮。")

    def test_gbk_text(self):
        parsed = extract_text("gbk.txt", "香云纱染整技艺".encode("gbk"))
        self.assertIn("香云纱", parsed.text)

    def test_blank_lines_collapsed(self):
        raw = "第一段\n\n\n\n第二段\n\n".encode("utf-8")
        self.assertEqual(extract_text("a.txt", raw).text, "第一段\n\n第二段")

    def test_empty_file_rejected(self):
        with self.assertRaises(UnsupportedDocument):
            extract_text("empty.txt", b"")

    def test_whitespace_only_rejected(self):
        with self.assertRaises(UnsupportedDocument):
            extract_text("blank.txt", "   \n\n  ".encode("utf-8"))


class DocxTests(unittest.TestCase):
    def test_paragraphs_and_tables_are_extracted(self):
        data = make_docx(
            ["陈家祠是广东民间工艺博物馆。", "馆内设有广绣、广彩展区。"],
            table_rows=[["项目", "级别"], ["广绣", "国家级"]],
        )
        parsed = extract_text("doc.docx", data)
        self.assertEqual(parsed.file_type, "Word")
        self.assertIn("陈家祠", parsed.text)
        self.assertIn("广绣 | 国家级", parsed.text)

    def test_empty_docx_rejected(self):
        with self.assertRaises(UnsupportedDocument):
            extract_text("empty.docx", make_docx([]))


class PdfTests(unittest.TestCase):
    def test_scanned_pdf_without_text_is_reported(self):
        with self.assertRaises(UnsupportedDocument) as ctx:
            extract_text("scan.pdf", make_blank_pdf())
        self.assertIn("OCR", str(ctx.exception))

    def test_broken_pdf_rejected(self):
        with self.assertRaises(UnsupportedDocument):
            extract_text("broken.pdf", b"not a real pdf")


class FormatGateTests(unittest.TestCase):
    def test_legacy_doc_has_clear_hint(self):
        with self.assertRaises(UnsupportedDocument) as ctx:
            extract_text("old.doc", b"whatever")
        self.assertIn(".docx", str(ctx.exception))

    def test_unknown_extension_rejected(self):
        with self.assertRaises(UnsupportedDocument):
            extract_text("data.xlsx", b"123")

    def test_supported_lists_are_consistent(self):
        self.assertEqual(sorted(UPLOAD_TYPES), sorted(["pdf", "docx", "txt"]))
        self.assertEqual(set(SUPPORTED_EXTENSIONS.values()), {"PDF", "Word", "TXT"})


if __name__ == "__main__":
    unittest.main()
