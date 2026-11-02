# -*- coding: utf-8 -*-
"""生成岭南非遗语料三份文档（txt / docx / pdf）。

用法：
    .venv\\Scripts\\python.exe tools/build_corpus.py txt,docx
    <含 reportlab 的 python> tools/build_corpus.py pdf

三种格式共用同一份内容与同一套行结构，保证导入知识库后切分结果一致。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from corpus_data import DOCUMENTS  # noqa: E402

OUT_DIR = os.path.join(ROOT, "assets", "corpus")


def blocks(doc):
    """统一的块序列：(类型, 文本)。"""
    rows = [("title", doc["title"])]
    rows += [("para", line) for line in doc["intro"]]
    for section, entries in doc["sections"]:
        rows.append(("section", section))
        for index, entry in enumerate(entries, start=1):
            rows.append(("entry", f"{index:02d}｜{entry['title']}"))
            rows += [("field", f"{name}：{value}") for name, value in entry["fields"]]
            rows.append(("blank", ""))
    return rows


def write_txt(doc, path):
    lines = []
    for kind, text in blocks(doc):
        if kind == "blank":
            lines.append("")
        elif kind == "section":
            lines.append("")
            lines.append("=" * 50)
            lines.append(text)
            lines.append("=" * 50)
        else:
            lines.append(text)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")


def write_docx(doc, path):
    import docx
    from docx.oxml.ns import qn
    from docx.shared import Pt

    document = docx.Document()

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing = 1.5

    for kind, text in blocks(doc):
        if kind == "blank":
            continue
        if kind == "title":
            document.add_heading(text, level=0)
        elif kind == "section":
            document.add_heading(text, level=1)
        elif kind == "entry":
            paragraph = document.add_paragraph()
            run = paragraph.add_run(text)
            run.bold = True
        else:
            document.add_paragraph(text)

    document.save(path)


def write_pdf(doc, path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    title_style = ParagraphStyle(
        "title", fontName="STSong-Light", fontSize=17, leading=26, spaceAfter=14
    )
    section_style = ParagraphStyle(
        "section", fontName="STSong-Light", fontSize=13, leading=22,
        spaceBefore=14, spaceAfter=8,
    )
    entry_style = ParagraphStyle(
        "entry", fontName="STSong-Light", fontSize=11, leading=20,
        spaceBefore=8, spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "body", fontName="STSong-Light", fontSize=10.5, leading=17,
        firstLineIndent=21, spaceAfter=2,
    )
    field_style = ParagraphStyle(
        "field", fontName="STSong-Light", fontSize=10.5, leading=17, spaceAfter=1
    )

    def esc(text):
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    story = []
    for kind, text in blocks(doc):
        if kind == "blank":
            story.append(Spacer(1, 6))
        elif kind == "title":
            story.append(Paragraph(esc(text), title_style))
        elif kind == "section":
            story.append(Paragraph(esc(text), section_style))
        elif kind == "entry":
            story.append(Paragraph(f"<b>{esc(text)}</b>", entry_style))
        elif kind == "para":
            story.append(Paragraph(esc(text), body_style))
        else:
            story.append(Paragraph(esc(text), field_style))

    SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=2.4 * cm, rightMargin=2.4 * cm,
        topMargin=2.2 * cm, bottomMargin=2.2 * cm,
        title=doc["title"],
    ).build(story)


WRITERS = {"txt": write_txt, "docx": write_docx, "pdf": write_pdf}


def main():
    formats = (sys.argv[1] if len(sys.argv) > 1 else "txt").split(",")
    os.makedirs(OUT_DIR, exist_ok=True)
    for doc in DOCUMENTS:
        for fmt in formats:
            name = os.path.splitext(doc["file"])[0] + "." + fmt
            path = os.path.join(OUT_DIR, name)
            WRITERS[fmt](doc, path)
            print(f"{fmt:>4}  {os.path.basename(path)}  {os.path.getsize(path):>7} bytes")


if __name__ == "__main__":
    main()
