"""配置文件：API key 和模型设置.

真实 API Key 请放在项目根目录 .env 中，不要提交到 git。
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ===== Embedding 配置（本地 bge 模型，不依赖 API）=====
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 512

# ===== LLM 生成配置 =====
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4.7-flash")

# ===== 路径配置 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ===== RAG 参数 =====
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 5
MAX_KNOWLEDGE_ITEMS = 20000  # 2万条（10倍于原demo）
