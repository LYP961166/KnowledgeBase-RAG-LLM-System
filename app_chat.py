"""Streamlit 知识库问答页：支持多会话管理与历史对话回溯。

会话与消息统一存放在 SQLite（storage.py），界面只做渲染与交互，
因此关闭浏览器、重启服务后重新进入仍能看到此前的会话记录。
"""

import streamlit as st

import config_data as config
from rag import RagService
from storage import ConversationStore

st.set_page_config(page_title="岭南非遗知识问答", page_icon="🏮", layout="centered")


@st.cache_resource
def get_store() -> ConversationStore:
    return ConversationStore(config.db_path)


@st.cache_resource
def get_rag() -> RagService:
    return RagService()


store = get_store()
rag = get_rag()


def switch_to(conversation_id: str) -> None:
    st.session_state.conversation_id = conversation_id
    st.query_params["c"] = conversation_id


# ---------------- 解析当前会话 ----------------
conversations = store.list_conversations()
requested = st.query_params.get("c") or st.session_state.get("conversation_id")
current = store.get_conversation(requested) if requested else None

if current is None:
    if conversations:
        current = conversations[0]
    else:
        current = store.get_conversation(store.create_conversation())
        conversations = store.list_conversations()

st.session_state.conversation_id = current["id"]
if st.query_params.get("c") != current["id"]:
    st.query_params["c"] = current["id"]


# ---------------- 侧边栏：会话管理 ----------------
with st.sidebar:
    st.subheader("💬 会话")
    if st.button("＋ 新建对话", type="primary", use_container_width=True):
        switch_to(store.create_conversation())
        st.rerun()

    st.caption(f"共 {len(conversations)} 个会话，点击切换")

    for conv in conversations:
        is_active = conv["id"] == current["id"]
        prefix = "▶ " if is_active else ""
        if st.button(
            f"{prefix}{conv['title']}",
            key=f"conv_{conv['id']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            help=f"{conv['message_count']} 条消息 · 最后更新 {conv['updated_at']}",
        ):
            switch_to(conv["id"])
            st.rerun()

    st.divider()
    with st.expander("管理当前会话"):
        new_title = st.text_input(
            "会话名称", value=current["title"], key=f"title_{current['id']}"
        )
        if st.button("保存名称", use_container_width=True):
            if store.rename_conversation(current["id"], new_title):
                st.rerun()
            else:
                st.warning("会话名称不能为空。")
        if st.button("删除此会话", use_container_width=True):
            store.delete_conversation(current["id"])
            st.session_state.pop("conversation_id", None)
            st.query_params.clear()
            st.rerun()


# ---------------- 主区 ----------------
st.title("🏮 岭南非遗文化景点知识问答")
st.caption("支持自然对话与多轮追问；知识类回答仅依据已录入知识库，并展示命中的参考来源。")
st.caption(f"当前会话：**{current['title']}**")

if rag.has_knowledge():
    st.success("依据知识库内容回答")
else:
    st.warning("知识库暂无数据，请先在知识库管理页面上传资料。")

st.divider()

messages = store.get_messages(current["id"])
if not messages:
    st.info("这是一个新会话，在下方输入问题即可开始对话。")

for message in messages:
    is_user = message["role"] == "user"
    with st.chat_message("user" if is_user else "assistant"):
        st.markdown(message["content"])

        sources = message.get("sources") or []
        if sources:
            with st.expander(f"查看参考来源（{len(sources)} 条）"):
                if message.get("search_query"):
                    st.caption(f"上下文检索问题：{message['search_query']}")
                for source in sources:
                    st.markdown(
                        f"**{source.get('source', '未知文档')}** · "
                        f"相关度 {float(source.get('score', 0)):.3f}"
                    )
                    st.caption(source.get("preview", ""))

        if not is_user:
            col_up, col_down, col_note = st.columns([1, 1, 6])
            if col_up.button("👍", key=f"up_{message['id']}"):
                store.set_feedback(message["id"], 1)
                st.rerun()
            if col_down.button("👎", key=f"down_{message['id']}"):
                store.set_feedback(message["id"], -1)
                st.rerun()
            if message.get("feedback", 0) > 0:
                col_note.caption("已标记：有用")
            elif message.get("feedback", 0) < 0:
                col_note.caption("已标记：无用")


# ---------------- 提问 ----------------
prompt = st.chat_input("请输入关于岭南非遗文化景点的问题")
if prompt:
    recent = store.get_messages(current["id"], limit=config.router_history_messages)

    try:
        with st.spinner("正在理解问题……"):
            route = rag.route_question(prompt, recent)
    except Exception as exc:
        print(f"Intent routing failed: {type(exc).__name__}: {exc}")
        store.add_message(current["id"], "user", prompt)
        store.add_message(current["id"], "assistant", "本次问题理解失败，请稍后重试。")
        st.rerun()

    intent = route["intent"]

    if intent in {"conversation", "unrelated"}:
        store.add_exchange(current["id"], prompt, route["reply"], intent=intent)
        st.rerun()

    if not rag.has_knowledge():
        store.add_exchange(
            current["id"],
            prompt,
            "知识库暂无数据，请先在知识库管理页面上传资料后再提问。",
            intent="empty",
        )
        st.rerun()

    results = rag.retrieve(route["search_query"])
    if not results:
        store.add_exchange(
            current["id"],
            prompt,
            "知识库未检索到相关信息，请提问岭南非遗文化景点相关内容。",
            intent="no_hit",
        )
        st.rerun()

    # 命中知识库：用户消息与回答由生成链写入历史，这里只负责流式展示与补写引用来源。
    search_query = route["search_query"]
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        try:
            for chunk in rag.stream_answer(prompt, results, current["id"]):
                answer += chunk if isinstance(chunk, str) else str(chunk)
                placeholder.markdown(answer)
        except Exception as exc:
            print(f"RAG request failed: {type(exc).__name__}: {exc}")
            # 生成链可能在失败前已写入本轮提问，这里避免重复插入。
            last = store.get_messages(current["id"], limit=1)
            already_logged = (
                last and last[-1]["role"] == "user" and last[-1]["content"] == prompt
            )
            if not already_logged:
                store.add_message(current["id"], "user", prompt)
            store.add_message(current["id"], "assistant", "本次回答生成失败，请稍后重试。")
            st.rerun()

    sources = [
        {
            "source": document.metadata.get("source", "未知文档"),
            "score": float(score),
            "preview": document.page_content[: config.source_preview_chars],
        }
        for document, score in results
    ]
    store.set_last_assistant_meta(
        current["id"], sources=sources, search_query=search_query, intent=intent
    )
    st.rerun()
