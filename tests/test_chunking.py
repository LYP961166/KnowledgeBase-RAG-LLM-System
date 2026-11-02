"""结构化切分测试：条目完整性、标题前缀、超长兜底与退化路径。"""

import sys
import unittest
from pathlib import Path

test_dir = Path(__file__).resolve().parent
project_root = test_dir if (test_dir / "chunking.py").exists() else test_dir.parent
sys.path.insert(0, str(project_root))

import config_data as config
import chunking
from chunking import split_document

SAMPLE = """岭南非遗文化景点资料

一、广东地区
==================================================
01｜陈家祠（广东民间工艺博物馆）
所在地区：广东省广州市荔湾区
景点类型：古建筑、民间工艺专题博物馆
核心非遗：广绣、广彩、广州玉雕、广州榄雕、石湾陶塑
参观提示：建筑细节丰富，建议结合讲解参观。

02｜永庆坊—粤剧艺术博物馆片区
所在地区：广东省广州市荔湾区恩宁路一带
景点类型：历史文化街区、戏曲专题博物馆
核心非遗：粤剧、粤曲、广彩、广绣、醒狮

03｜广东省非物质文化遗产馆
所在地区：广东省佛山市顺德区
景点类型：省级综合性非遗场馆
核心非遗：粤剧、广东音乐、岭南传统舞蹈与戏曲

二、广西地区
==================================================
01｜广西民族博物馆
所在地区：广西壮族自治区南宁市青秀区
核心非遗：壮族织锦技艺、侗族大歌
"""

PLAIN = (
    "岭南地区的非物质文化遗产资源十分丰富，涵盖传统音乐、传统舞蹈、传统戏剧等多个门类。"
    "这些遗产既是中华优秀传统文化的重要组成部分，也是区域文化认同的重要载体。"
) * 20


class StructuredSplitTests(unittest.TestCase):
    def setUp(self):
        self._original_mode = config.chunking_mode
        config.chunking_mode = "structured"

    def tearDown(self):
        config.chunking_mode = self._original_mode

    def test_one_entry_per_chunk(self):
        chunks = split_document(SAMPLE)
        titles = [chunk.title for chunk in chunks]
        self.assertIn("陈家祠（广东民间工艺博物馆）", titles)
        self.assertIn("永庆坊—粤剧艺术博物馆片区", titles)
        self.assertIn("广西民族博物馆", titles)

    def test_entries_are_not_mixed(self):
        chunks = split_document(SAMPLE)
        chen_clan = next(c for c in chunks if c.title.startswith("陈家祠"))
        self.assertIn("广绣", chen_clan.text)
        self.assertNotIn("永庆坊", chen_clan.text)
        self.assertNotIn("广西民族博物馆", chen_clan.text)

    def test_header_carries_section_and_title(self):
        chunks = split_document(SAMPLE)
        chen_clan = next(c for c in chunks if c.title.startswith("陈家祠"))
        self.assertEqual(chen_clan.section, "广东地区")
        self.assertTrue(chen_clan.text.startswith("【广东地区·陈家祠（广东民间工艺博物馆）】"))

    def test_second_section_has_its_own_name(self):
        chunks = split_document(SAMPLE)
        guangxi = next(c for c in chunks if c.title == "广西民族博物馆")
        self.assertEqual(guangxi.section, "广西地区")
        self.assertIn("【广西地区·广西民族博物馆】", guangxi.text)

    def test_separator_lines_are_dropped(self):
        chunks = split_document(SAMPLE)
        for chunk in chunks:
            self.assertNotIn("====", chunk.text)

    def test_long_entry_is_split_and_keeps_header(self):
        body = "".join(f"第{i}条说明：该场馆展示岭南传统技艺。\n" for i in range(120))
        text = f"一、广东地区\n01｜超长条目\n{body}"
        chunks = split_document(text, chunk_size=300, chunk_overlap=0)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.text.startswith("【广东地区·超长条目】"))
            self.assertLessEqual(len(chunk.text), 320)

    def test_fallback_when_no_structure(self):
        chunks = split_document(PLAIN, chunk_size=300, chunk_overlap=50)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk.text) <= 300 for chunk in chunks))
        self.assertTrue(all(chunk.title == "" for chunk in chunks))

    def test_fixed_mode_ignores_structure(self):
        config.chunking_mode = "fixed"
        chunks = split_document(SAMPLE, chunk_size=200, chunk_overlap=40)
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.title == "" for chunk in chunks))

    def test_empty_text_returns_nothing(self):
        self.assertEqual(split_document(""), [])
        self.assertEqual(split_document("   \n\n  "), [])


if __name__ == "__main__":
    unittest.main()
