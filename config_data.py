"""Application configuration shared by the upload and chat services."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Anchor persistence to the project instead of the terminal working directory.
persist_directory = str(BASE_DIR / "chroma_db")
collection_name = "rag"

# 会话与消息记录（SQLite），随项目目录一起移动。
db_path = str(BASE_DIR / "data" / "app.db")
default_conversation_title = "新对话"
conversation_title_chars = 20

chunk_size = 700
chunk_overlap = 120
separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

# 切分策略：structured=按文档层级（章/条目）切分，fixed=固定长度递归切分。
# 保留 fixed 用于对照实验；切换后需清空知识库重新上传，否则新旧片段会混在一起。
chunking_mode = "structured"

# 向量库距离度量。Chroma 只在创建集合时确定，改这里必须重置知识库。
distance_metric = "cosine"

# 检索：两路各召回若干，RRF 融合后送重排，最终取 top_k 条进提示词。
retrieval_top_k = 4
vector_candidates = 10
keyword_candidates = 10
rrf_k = 60
rerank_candidates = 8
rerank_enabled = True
rerank_model_name = "gte-rerank-v2"
# 重排分数低于该值的片段不进入提示词；全部被过滤时保留重排结果的最前一条。
rerank_min_score = 0.05

embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen-plus"

max_history_messages = 12
max_ui_messages = 40
# Recent UI messages supplied to the intent router for contextual follow-ups.
router_history_messages = 8
chat_request_timeout_seconds = 120
chat_max_retries = 2
source_preview_chars = 360
