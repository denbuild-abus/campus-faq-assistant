"""索引持久化：把检索器的文本块、元数据、向量落盘，启动时恢复。"""
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.retriever import HybridRetriever, tokenize


class IndexStore:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return (self.index_dir / "embeddings.npz").exists()

    def save(self, retriever: HybridRetriever) -> None:
        np.savez(self.index_dir / "embeddings.npz", embeddings=retriever.embeddings)
        (self.index_dir / "chunks.json").write_text(
            json.dumps(retriever.chunks, ensure_ascii=False), encoding="utf-8"
        )
        (self.index_dir / "metadata.json").write_text(
            json.dumps(retriever.metadata, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, retriever: HybridRetriever) -> HybridRetriever:
        data = np.load(self.index_dir / "embeddings.npz")
        retriever.embeddings = data["embeddings"]
        retriever.chunks = json.loads(
            (self.index_dir / "chunks.json").read_text(encoding="utf-8")
        )
        retriever.metadata = json.loads(
            (self.index_dir / "metadata.json").read_text(encoding="utf-8")
        )
        retriever.tokenized = [tokenize(c) for c in retriever.chunks]
        retriever.bm25 = BM25Okapi(retriever.tokenized)
        return retriever
