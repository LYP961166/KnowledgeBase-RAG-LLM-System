# -*- coding: utf-8 -*-
"""切分策略对照实验：fixed（固定长度） vs structured（按文档层级）。

用法：
    .venv\\Scripts\\python.exe tools/compare_chunking.py [语料路径]

输出可直接作为设计说明书中「切分策略对比」一节的数据来源。
"""

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config_data as config
import chunking

ENTRY_RE = re.compile(r"^\d{2,}｜\s*(.+)$", re.M)

DEFAULT_CORPUS = ROOT / "assets" / "岭南非遗文化景点知识库资料.txt"


def measure(text: str, mode: str) -> dict:
    original = config.chunking_mode
    config.chunking_mode = mode
    try:
        chunks = chunking.split_document(text, config.chunk_size, config.chunk_overlap)
    finally:
        config.chunking_mode = original

    lengths = [len(chunk.text) for chunk in chunks]
    multi = sum(1 for chunk in chunks if len(ENTRY_RE.findall(chunk.text)) >= 2)

    # 条目完整性：首个字段行与最后一个字段行是否落在同一片段。
    # 比"字符串是否重复出现"更可靠，两种模式可用同一把尺子。
    blocks = [(s, t, b) for s, t, b in chunking._parse_blocks(chunking.normalize(text)) if b]
    intact = 0
    for _, _, body in blocks:
        head, tail = body[0][:15], body[-1][:15]
        if any(head in chunk.text and tail in chunk.text for chunk in chunks):
            intact += 1

    tagged = sum(1 for chunk in chunks if chunk.section or chunk.title)
    return {
        "mode": mode,
        "chunks": len(chunks),
        "avg": sum(lengths) / len(lengths) if lengths else 0,
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "multi": multi,
        "blocks": len(blocks),
        "intact": intact,
        "tagged": tagged,
    }


def main() -> None:
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CORPUS
    text = corpus.read_text(encoding="utf-8")
    print(f"语料：{corpus.name}（{len(text)} 字，{len(ENTRY_RE.findall(text))} 个条目）")
    print(f"参数：chunk_size={config.chunk_size} chunk_overlap={config.chunk_overlap}\n")

    rows = [measure(text, "fixed"), measure(text, "structured")]
    header = f"{'指标':<20}{'fixed':>14}{'structured':>14}"
    print(header)
    print("-" * 50)
    for label, key, fmt in [
        ("片段总数", "chunks", "d"),
        ("平均长度", "avg", ".0f"),
        ("最短 / 最长", None, None),
        ("混入多个条目的片段", "multi", "d"),
        ("条目完整保留", "intact", "d"),
        ("带章节元数据的片段", "tagged", "d"),
    ]:
        if key is None:
            print(f"{label:<20}{rows[0]['min']:>6}/{rows[0]['max']:<7}"
                  f"{rows[1]['min']:>6}/{rows[1]['max']:<7}")
            continue
        left = format(rows[0][key], fmt)
        right = format(rows[1][key], fmt)
        print(f"{label:<20}{left:>14}{right:>14}")

    base, new = rows
    print(f"\n条目总数：{base['blocks']}")
    print(f"结论：混入多个条目的片段 {base['multi']} → {new['multi']}；"
          f"条目完整保留 {base['intact']}/{base['blocks']} → {new['intact']}/{new['blocks']}。")


if __name__ == "__main__":
    main()
