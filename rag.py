"""Retrieval-augmented generation service."""

import json
import re

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

import config_data as config
from sqlite_history_store import get_history
from vector_stores import VectorStoreService


VALID_INTENTS = {"conversation", "knowledge", "unrelated"}
ROUTER_OUTPUT_SCHEMA = (
    '{{"intent":"conversation|knowledge|unrelated",'
    '"reply":"conversation 或 unrelated 的动态回复",'
    '"search_query":"仅 knowledge 使用的上下文完整检索问题"}}'
)


def parse_route_response(raw_response: str) -> dict[str, str]:
    """Parse the router's JSON, accepting optional Markdown code fences."""
    text = raw_response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    json_object = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not json_object:
        raise ValueError("Router did not return a JSON object")
    data = json.loads(json_object.group(0))
    intent = str(data.get("intent", "")).strip().lower()
    if intent not in VALID_INTENTS:
        raise ValueError(f"Unsupported router intent: {intent}")
    reply = str(data.get("reply") or "").strip()
    search_query = str(data.get("search_query") or "").strip()
    if intent in {"conversation", "unrelated"} and not reply:
        raise ValueError(f"{intent} route did not include a reply")
    if intent == "knowledge" and not search_query:
        raise ValueError("Knowledge route did not include a search query")
    return {"intent": intent, "reply": reply, "search_query": search_query}


class RagService:
    def __init__(self) -> None:
        self.vector_service = VectorStoreService()
        self.chain = None
        self.router_chain = None

    def _get_router_chain(self):
        if self.router_chain is not None:
            return self.router_chain
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是岭南非遗文化景点问答系统的意图路由器，同时负责自然对话回复。\n"
                    "请将当前输入严格分为以下一种意图：\n"
                    "- conversation：问候、致谢、告别、询问助手身份或能力、对当前对话的自然交流。"
                    "这类输入需要以岭南非遗知识助手身份自然回复。\n"
                    "- knowledge：询问岭南非物质文化遗产、文化景点、场馆、传统技艺、地区文化，"
                    "或基于最近对话继续追问这些内容。必须结合最近对话，把省略主语、指代词或模糊追问"
                    "改写成一个脱离上下文也能理解的完整 search_query。"
                    "例如上轮询问深圳景点，本轮问‘那广州呢’，search_query 应改写为"
                    "‘广州有哪些值得参观的岭南非遗文化景点’。\n"
                    "- unrelated：天气、编程、数学、娱乐等与岭南非遗知识及当前对话无关的请求。"
                    "这类输入不能回答其具体内容；需要礼貌说明系统专注于岭南非遗文化景点，"
                    "并结合系统能力给出 2 至 3 个简短、具体、可直接提问的问题示例。"
                    "示例尽量根据最近对话变化，不要机械重复。\n\n"
                    "knowledge 的 reply 必须为空字符串且 search_query 必须非空；"
                    "conversation 和 unrelated 必须提供自然、友好的 reply，search_query 必须为空字符串。\n"
                    "只输出一个 JSON 对象，不要输出 Markdown 或解释：\n"
                    + ROUTER_OUTPUT_SCHEMA,
                ),
                ("user", "最近对话：\n{history}\n\n当前用户输入：{input}"),
            ]
        )
        model = ChatTongyi(
            model=config.chat_model_name,
            top_p=0.1,
            max_retries=config.chat_max_retries,
            model_kwargs={"request_timeout": config.chat_request_timeout_seconds},
        )
        self.router_chain = prompt | model | StrOutputParser()
        return self.router_chain

    def _get_chain(self):
        if self.chain is not None:
            return self.chain
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是岭南非遗文化景点知识库问答助手。你必须遵守以下规则：\n"
                    "1. 仅依据给出的参考资料回答，不使用参考资料之外的事实补全答案。\n"
                    "2. 资料不足时明确说‘当前知识库未找到足够可靠的资料’，不得猜测。\n"
                    "3. 涉及开放时间、票价、预约、交通和临时活动时，提醒用户以场馆最新官方公告为准。\n"
                    "4. 回答简洁、准确，并优先给出与问题直接相关的信息。\n\n"
                    "参考资料：\n{context}",
                ),
                MessagesPlaceholder("history"),
                ("user", "{input}"),
            ]
        )
        model = ChatTongyi(
            model=config.chat_model_name,
            max_retries=config.chat_max_retries,
            model_kwargs={"request_timeout": config.chat_request_timeout_seconds},
        )
        self.chain = RunnableWithMessageHistory(
            prompt | model | StrOutputParser(),
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return self.chain

    def has_knowledge(self) -> bool:
        return self.vector_service.count() > 0

    def route_question(self, question: str, recent_messages: list[dict]) -> dict[str, str]:
        history_lines = []
        for message in recent_messages[-config.router_history_messages :]:
            role = "用户" if message.get("role") == "user" else "助手"
            content = str(message.get("content") or "")[:500]
            history_lines.append(f"{role}：{content}")
        history = "\n".join(history_lines) or "无"
        raw_response = self._get_router_chain().invoke(
            {"input": question, "history": history}
        )
        return parse_route_response(raw_response)

    def retrieve(self, question: str):
        return self.vector_service.search(question)

    @staticmethod
    def format_context(results) -> str:
        sections = []
        for index, (document, score) in enumerate(results, start=1):
            source = document.metadata.get("source", "未知文档")
            # 带上所属章节，便于模型回答时说明"这是哪个场馆/哪个地区的资料"
            section = document.metadata.get("section") or ""
            title = document.metadata.get("section_title") or ""
            origin = "｜".join(part for part in (source, section, title) if part)
            sections.append(
                f"[{index}] 来源：{origin}｜相关度：{score:.3f}\n{document.page_content}"
            )
        return "\n\n".join(sections)

    def stream_answer(self, question: str, results, session_id: str):
        session_config = {"configurable": {"session_id": session_id}}
        return self._get_chain().stream(
            {"input": question, "context": self.format_context(results)},
            session_config,
        )
