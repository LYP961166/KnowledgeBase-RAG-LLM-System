"""Streamlit 知识库管理页：上传、查看与删除知识文档。

支持 TXT / PDF / Word 三种格式，文件解析交给 document_loader，
本页只负责交互与展示。
"""

import html
import os
import re
import uuid
from pathlib import Path

import streamlit as st

import config_data as config
from document_loader import (
    SUPPORTED_EXTENSIONS,
    UnsupportedDocument,
    extract_text,
)
from knowledge_base import KnowledgeBaseService

st.set_page_config(page_title="岭南非遗知识库管理", page_icon="📚", layout="wide")

TYPE_ICONS = {"PDF": "📕", "Word": "📘", "TXT": "📄"}
EXT_TYPES = {".pdf": "PDF", ".docx": "Word", ".doc": "Word", ".txt": "TXT"}

CSS = """
<style>
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1180px; }
.kb-hero {
    border: 1px solid rgba(15,110,86,0.16);
    background: #F4FAF7;
    border-radius: 14px;
    padding: 20px 26px 18px 26px;
    margin-bottom: 20px;
}
.kb-hero h1 { font-size: 23px; margin: 0 0 8px 0; color: #0B4F3D; font-weight: 600; }
.kb-hero p { margin: 0; color: #4B6B62; font-size: 13.5px; line-height: 1.75; }
.kb-tag {
    display: inline-block; padding: 2px 11px; border-radius: 999px;
    background: #E1F5EE; color: #0F6E56; font-size: 12px;
    margin: 8px 6px 0 0; border: 1px solid rgba(15,110,86,0.18);
}
.kb-empty { color: #8A8A8A; font-size: 14px; padding: 22px 0; text-align: center; }
.kb-hint { color: #6B7C76; font-size: 13px; line-height: 1.9; }

/* 成功提示：完整显示 5 秒后瞬间消失，不留空位 */
.kb-flash {
    background: #E1F5EE;
    border: 1px solid rgba(15,110,86,0.25);
    border-left: 3px solid #0F6E56;
    color: #0B4F3D;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13.5px;
    line-height: 1.7;
    margin-bottom: 8px;
    overflow: hidden;
    animation: kb-flash-out 5s linear forwards;
}
@keyframes kb-flash-out {
    0%, 99.99% { opacity: 1; max-height: 200px; }
    100% { opacity: 0; max-height: 0; padding-top: 0; padding-bottom: 0;
           margin-bottom: 0; border-width: 0; }
}

/* 警告与错误：常驻，可手动关闭 */
.kb-notice {
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13.5px;
    line-height: 1.7;
    margin-bottom: 6px;
}
.kb-notice.is-warning {
    background: #FDF6E3;
    border: 1px solid rgba(186,117,23,0.28);
    border-left: 3px solid #BA7517;
    color: #6B4A0F;
}
.kb-notice.is-error {
    background: #FDEDED;
    border: 1px solid rgba(196,60,60,0.28);
    border-left: 3px solid #C43C3C;
    color: #7A2020;
}

/* 上传区：加大拖拽热区，并把自带的英文文案换成中文
   该容器实际是 section，且自带样式特异性较高，故统一用 !important 覆盖 */
[data-testid="stFileUploaderDropzone"] {
    min-height: 130px !important;
    border: 1.5px dashed rgba(15,110,86,0.35) !important;
    background: #F7FCFA !important;
    border-radius: 12px !important;
    padding: 18px 16px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div:last-child {
    font-size: 0 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small {
    font-size: 0 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small::after {
    content: "单个文件不超过 200MB" !important;
    font-size: 12px !important;
    color: #8A9A94 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div:last-child::after {
    content: "把文件拖到这里，或点右侧按钮选择" !important;
    font-size: 13.5px !important;
    color: #4B6B62 !important;
    line-height: 1.8 !important;
}
[data-testid="stFileUploaderDropzone"] button {
    position: relative !important;
    min-width: 96px !important;
    min-height: 36px !important;
    font-size: 0 !important;
}
[data-testid="stFileUploaderDropzone"] button::after {
    content: "选择文件" !important;
    position: absolute !important;
    inset: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 13.5px !important;
    color: inherit !important;
}
</style>
"""


def safe_error_detail(exc: Exception) -> str:
    """返回可展示的诊断信息，同时抹掉可能出现的密钥。"""
    detail = str(exc).strip().replace("\r", " ").replace("\n", " ")
    detail = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "***", detail)
    detail = re.sub(r"(?i)(api[_ -]?key\s*[:=]\s*)[^,;\s]+", r"\1***", detail)
    return detail[:400] or "未返回详细错误信息"


def failure_reason(exc: Exception) -> str:
    text = str(exc).lower()
    if "api_key" in text or "status_code: 401" in text:
        return "DashScope Key 缺失或无效"
    if "model not exist" in text:
        return "Embedding 模型不可用"
    if "timeout" in text or "connection" in text:
        return "网络连接或请求超时"
    return type(exc).__name__


st.markdown(CSS, unsafe_allow_html=True)

if "kb_service" not in st.session_state:
    st.session_state.kb_service = KnowledgeBaseService()
service = st.session_state.kb_service

# uploader_round 变化即重建上传控件，用来在录入完成后清空已选文件
st.session_state.setdefault("uploader_round", 0)
st.session_state.setdefault("kb_notices", [])


def collect_notice(bucket: list, level: str, message: str) -> None:
    """同一批里相同内容只保留一条，避免重复上传时刷屏。"""
    if any(item["level"] == level and item["message"] == message for item in bucket):
        return
    bucket.append({"id": uuid.uuid4().hex[:8], "level": level, "message": message})


def render_notices() -> None:
    """成功提示只渲染一次（CSS 控制 5 秒后消失），警告与错误常驻且可逐条关闭。"""
    flashes = st.session_state.pop("kb_flash", [])
    if flashes:
        items = "".join(f'<div class="kb-flash">{html.escape(text)}</div>' for text in flashes)
        st.markdown(items, unsafe_allow_html=True)

    notices = st.session_state.get("kb_notices", [])
    for notice in list(notices):
        text_col, close_col = st.columns([24, 2], vertical_alignment="center")
        with text_col:
            st.markdown(
                f'<div class="kb-notice is-{notice["level"]}">'
                f"{html.escape(notice['message'])}</div>",
                unsafe_allow_html=True,
            )
        with close_col:
            if st.button("✕", key=f"dismiss_{notice['id']}", help="移除这条提示"):
                st.session_state.kb_notices = [
                    item for item in notices if item["id"] != notice["id"]
                ]
                st.rerun()


render_notices()

# ---------------- 页头 ----------------
tags = "".join(f'<span class="kb-tag">{name}</span>' for name in SUPPORTED_EXTENSIONS.values())
st.markdown(
    '<div class="kb-hero">'
    "<h1>📚 岭南非遗知识库</h1>"
    "<p>上传文档后自动完成解析、切分、去重与向量化，问答页立即生效。</p>"
    f"<p>支持格式：{tags}</p>"
    "</div>",
    unsafe_allow_html=True,
)

status = service.get_status()
average = status["chunk_count"] / status["document_count"] if status["document_count"] else 0
metric_left, metric_mid, metric_right = st.columns(3)
metric_left.metric("已录入文档", status["document_count"])
metric_mid.metric("知识片段", status["chunk_count"])
metric_right.metric("平均片段 / 篇", f"{average:.1f}")

st.write("")

if service.needs_rebuild():
    stored = service.read_build_info() or {}
    st.warning(
        "当前知识库是用旧参数建立的"
        f"（度量 `{stored.get('distance_metric', '未记录')}`、"
        f"切分 `{stored.get('chunking_mode', '未记录')}`），"
        f"与当前配置（`{config.distance_metric}` / `{config.chunking_mode}`）不一致，"
        "检索结果会不准确。请在页面下方「维护：重置知识库」重建后重新上传。"
    )

# ---------------- 上传区 ----------------
api_key_ready = bool(os.getenv("DASHSCOPE_API_KEY", "").strip())
upload_column, hint_column = st.columns([1.6, 1], gap="large")

with upload_column:
    with st.container(border=True):
        st.markdown("#### 上传文档")
        st.caption("支持拖拽上传，也可点击选择；一次可选多个文件。")
        # 不限制 type：交给 Streamlit 过滤会弹出英文提示，这里统一在解析阶段用中文报错
        uploaded_files = st.file_uploader(
            "选择文件",
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"kb_uploader_{st.session_state.uploader_round}",
            help="支持 PDF / Word(.docx) / TXT；其他格式会给出中文提示。",
        )

        if not api_key_ready:
            st.warning(
                "当前进程未检测到 DASHSCOPE_API_KEY，无法生成向量。"
                "请停止服务，在同一终端设置环境变量后重新启动。"
            )
            st.code(
                '$env:DASHSCOPE_API_KEY="你的新API_KEY"\n'
                '.\\.venv\\Scripts\\python.exe -m streamlit run app_upload.py --server.port 8501',
                language="powershell",
            )

        if st.button(
            "开始录入",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files or not api_key_ready,
        ):
            flashes: list[str] = []
            persistent: list[dict] = []
            repeated: list[str] = []
            total = len(uploaded_files)
            progress = st.progress(0.0, text="准备解析……")

            for index, uploaded_file in enumerate(uploaded_files, start=1):
                name = uploaded_file.name
                progress.progress(
                    (index - 1) / total, text=f"正在处理《{name}》（{index}/{total}）"
                )
                try:
                    parsed = extract_text(name, uploaded_file.getvalue())
                except UnsupportedDocument as exc:
                    collect_notice(persistent, "error", f"《{name}》{exc}")
                    continue
                except Exception as exc:
                    collect_notice(
                        persistent, "error", f"《{name}》解析失败：{safe_error_detail(exc)}"
                    )
                    continue

                try:
                    result = service.upload_by_str(parsed.text, name, parsed.file_type)
                except Exception as exc:
                    print(f"Knowledge ingestion failed: {type(exc).__name__}: {exc}")
                    collect_notice(
                        persistent,
                        "error",
                        f"《{name}》录入失败：{failure_reason(exc)}。"
                        f"技术详情：{safe_error_detail(exc)}",
                    )
                    continue

                if result["status"] == "success":
                    extra = f"，共 {parsed.pages} 页" if parsed.pages else ""
                    message = f"{result['message']}（{parsed.file_type}{extra}）"
                    if message not in flashes:
                        flashes.append(message)
                elif result["status"] == "repeat":
                    repeated.append(name)
                else:
                    collect_notice(persistent, "error", result["message"])

            # 重复内容合并成一条，避免逐文件刷屏
            if repeated:
                names = "、".join(f"《{item}》" for item in repeated)
                collect_notice(
                    persistent, "warning", f"{names} 的内容已存在于知识库，本次已跳过。"
                )

            progress.progress(1.0, text="处理完成")
            if not any(item["level"] == "error" for item in persistent):
                # 整批无错误才清空上传控件；有错误时保留文件，方便修正后重试
                st.session_state.uploader_round += 1
            st.session_state.kb_flash = flashes
            st.session_state.kb_notices = persistent
            if flashes or persistent:
                st.rerun()

with hint_column:
    with st.container(border=True):
        st.markdown("#### 上传说明")
        st.markdown(
            '<div class="kb-hint">'
            "• <b>PDF</b>：需可复制文字，扫描件请先做 OCR<br>"
            "• <b>Word</b>：仅支持 .docx，旧版 .doc 请先另存<br>"
            "• <b>TXT</b>：UTF-8 与 GBK 编码会自动识别<br>"
            "• 相同内容重复上传会被跳过<br>"
            "• 写入向量库需调用嵌入接口，篇幅越长耗时越久；<br>"
            "&nbsp;&nbsp;大文件建议拆分后再上传"
            "</div>",
            unsafe_allow_html=True,
        )

st.write("")

# ---------------- 文档列表 ----------------
st.markdown("#### 已录入文档")
documents = service.list_documents()

if not documents:
    st.markdown(
        '<div class="kb-empty">知识库暂无数据，请在上方上传文档。</div>',
        unsafe_allow_html=True,
    )
else:
    for item in documents:
        with st.container(border=True):
            name_col, chunk_col, time_col, action_col = st.columns([5.4, 1.1, 2.2, 1.0])
            # 旧数据的元数据里没有 file_type，按扩展名兜底显示
            shown_type = item["file_type"] or EXT_TYPES.get(
                Path(item["source"]).suffix.lower(), ""
            )
            icon = TYPE_ICONS.get(shown_type, "📄")
            name_col.markdown(f"{icon} **{item['source']}**")
            name_col.caption(shown_type or "未知格式")

            chunk_col.markdown(f"**{item['chunk_count']}**")
            chunk_col.caption("片段")

            created = item["create_time"] or "—"
            time_col.markdown(created[:10] if len(created) >= 10 else created)
            time_col.caption(created[11:16] if len(created) >= 16 else "")

            if item["file_hash"]:
                if action_col.button(
                    "删除", key=f"del_{item['file_hash']}", use_container_width=True
                ):
                    if service.delete_document(item["file_hash"]):
                        st.session_state.kb_flash = [f"已删除《{item['source']}》。"]
                        st.rerun()
                    else:
                        st.error("未找到该文档，请刷新页面后重试。")
            else:
                action_col.caption("—")

    if any(not item["file_hash"] for item in documents):
        st.warning("检测到旧版知识片段，请清空知识库后重新上传以启用按文档删除。")

    with st.expander("维护：重置知识库"):
        st.caption(
            "切分方式或距离度量改动后，必须重置知识库并重新上传，旧片段才会按新配置重建。"
        )
        confirmed = st.checkbox("我确认要删除全部知识文档并重建索引")
        if st.button("重置知识库", disabled=not confirmed):
            deleted = service.reset_collection()
            st.session_state.kb_flash = [
                f"已重置知识库，删除 {deleted} 个片段，请重新上传文档。"
            ]
            st.rerun()
