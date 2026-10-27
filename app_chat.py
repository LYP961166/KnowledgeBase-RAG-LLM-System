import streamlit as  st
from rag import RagService
import config_data as config

# 标题
st.title("智能客服")
# 分隔符
st.divider()

# 避免性能压力，session_state 存入对象
if "message" not in st.session_state:
    st.session_state["message"]=[{"role":"assistant","content":"你好，有什么可以帮助你？"}]

if (
    "rag" not in st.session_state
    or st.session_state.get("rag_model_name") != config.chat_model_name
):
    st.session_state["rag"]= RagService()
    st.session_state["rag_model_name"] = config.chat_model_name
#循环 输出历史信息，原本只记录但页面不显示
for message in st.session_state["message"][-config.max_ui_messages:]:
    st.chat_message(message["role"]).write(message["content"])
# 在页面最下方提供用户输入栏
prompt= st.chat_input()

if prompt :
    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})
    st.session_state["message"] = st.session_state["message"][-config.max_ui_messages:]

    ai_res_list = []
    with st.spinner("AI 思考中......."):
        # 保留流式输出，但显式更新占位区域；这样模型流中途出错时，
        # 可以在页面提示并结束本轮，不让异常直接破坏 Streamlit 连接。
        assistant_container = st.chat_message("assistant")
        response_placeholder = assistant_container.empty()
        response_text = ""

        try:
            res_stream = st.session_state["rag"].chain.stream(
                {"input": prompt}, config.session_config
            )
            for chunk in res_stream:
                text = chunk if isinstance(chunk, str) else str(chunk)
                ai_res_list.append(text)
                response_text += text
                response_placeholder.markdown(response_text)
        except Exception as exc:
            if response_text:
                response_placeholder.markdown(response_text)
            st.error("本次回答生成中断，请稍后重试。详细错误已记录在运行终端。")
            print(f"RAG request failed: {type(exc).__name__}: {exc}")
        else:
            st.session_state["message"].append(
                {"role": "assistant", "content": response_text}
            )
            st.session_state["message"] = st.session_state["message"][-config.max_ui_messages:]

