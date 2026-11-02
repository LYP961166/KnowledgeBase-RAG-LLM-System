import sys
import unittest
from pathlib import Path

test_dir = Path(__file__).resolve().parent
project_root = test_dir if (test_dir / "rag.py").exists() else test_dir.parent
sys.path.insert(0, str(project_root))

from langchain_core.prompts import ChatPromptTemplate

from rag import ROUTER_OUTPUT_SCHEMA, parse_route_response


class RouteResponseParserTests(unittest.TestCase):
    def test_router_json_schema_does_not_become_prompt_variable(self):
        prompt = ChatPromptTemplate.from_messages(
            [("system", ROUTER_OUTPUT_SCHEMA), ("user", "{input}")]
        )
        self.assertEqual(prompt.input_variables, ["input"])
        messages = prompt.format_messages(input="你好")
        self.assertIn('{"intent":', messages[0].content)

    def test_parses_conversation_reply(self):
        result = parse_route_response(
            '{"intent":"conversation","reply":"你好，我是岭南非遗知识助手。","search_query":""}'
        )
        self.assertEqual(result["intent"], "conversation")
        self.assertIn("岭南非遗", result["reply"])

    def test_parses_json_code_fence(self):
        result = parse_route_response(
            '```json\n{"intent":"knowledge","reply":"","search_query":"广州有哪些岭南非遗文化景点"}\n```'
        )
        self.assertEqual(result["intent"], "knowledge")
        self.assertIn("广州", result["search_query"])

    def test_parses_unrelated_route(self):
        result = parse_route_response(
            '{"intent":"unrelated","reply":"这个问题超出知识库范围。你可以问：陈家祠有哪些非遗项目？","search_query":""}'
        )
        self.assertEqual(result["intent"], "unrelated")
        self.assertIn("陈家祠", result["reply"])

    def test_rejects_unknown_intent(self):
        with self.assertRaises(ValueError):
            parse_route_response(
                '{"intent":"weather","reply":"","search_query":""}'
            )

    def test_conversation_requires_dynamic_reply(self):
        with self.assertRaises(ValueError):
            parse_route_response(
                '{"intent":"conversation","reply":"","search_query":""}'
            )

    def test_unrelated_requires_dynamic_guidance(self):
        with self.assertRaises(ValueError):
            parse_route_response(
                '{"intent":"unrelated","reply":"","search_query":""}'
            )

    def test_knowledge_requires_contextual_search_query(self):
        with self.assertRaises(ValueError):
            parse_route_response(
                '{"intent":"knowledge","reply":"","search_query":""}'
            )


if __name__ == "__main__":
    unittest.main()
