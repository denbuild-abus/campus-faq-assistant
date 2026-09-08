"""应用配置：从环境变量读取，带默认值。"""
import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # 大模型（兼容 OpenAI 协议：DeepSeek / 通义千问 / OpenAI / Moonshot 等）
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")

    # Embedding 模型（首次调用自动下载并缓存）
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    # 检索参数
    top_k: int = _env_int("TOP_K", 5)
    chunk_size: int = _env_int("CHUNK_SIZE", 256)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 32)

    # 数据目录
    docs_dir: str = os.getenv("DOCS_DIR", "data/docs")
    index_dir: str = os.getenv("INDEX_DIR", "data/index")


settings = Settings()
