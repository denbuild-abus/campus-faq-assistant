"""混合检索：BM25 关键词 + 向量语义，RRF 融合排序。"""
import re
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

try:
    import jieba

    _HAS_JIEBA = True
except ImportError:  # pragma: no cover
    _HAS_JIEBA = False


def tokenize(text: str) -> List[str]:
    """中文优先用 jieba 分词，英文/数字按词切，返回词元列表。"""
    tokens: List[str] = []
    for seg in re.split(r"(\s+)", text):
        seg = seg.strip()
        if not seg:
            continue
        if re.search(r"[一-鿿]", seg):
            tokens.extend(jieba.lcut(seg) if _HAS_JIEBA else list(seg))
        else:
            tokens.extend(re.findall(r"[A-Za-z0-9_]+", seg))
    return [t for t in tokens if t]


class HybridRetriever:
    """BM25（关键词精确匹配）+ 向量（语义相似）的混合检索器。"""

    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: List[str] = []
        self.metadata: List[dict] = []
        self.embeddings: np.ndarray = None
        self.tokenized: List[List[str]] = []
        self.bm25: BM25Okapi = None

    # ---- 构建 ----
    def build(self, chunks: List[str], metadata: List[dict]) -> None:
        self.chunks = list(chunks)
        self.metadata = list(metadata)
        self.embeddings = self.embedder.embed(chunks)
        self.tokenized = [tokenize(c) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    def add(self, new_chunks: List[str], new_metadata: List[dict]) -> None:
        """增量追加文档块并重建 BM25。"""
        if not new_chunks:
            return
        new_emb = self.embedder.embed(new_chunks)
        self.embeddings = (
            np.vstack([self.embeddings, new_emb])
            if self.embeddings is not None
            else new_emb
        )
        self.chunks.extend(new_chunks)
        self.metadata.extend(new_metadata)
        self.tokenized.extend([tokenize(c) for c in new_chunks])
        self.bm25 = BM25Okapi(self.tokenized)

    # ---- 检索 ----
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        if not self.chunks:
            return []

        n = len(self.chunks)
        q_tokens = tokenize(query)
        q_emb = self.embedder.embed(query)[0]

        bm25_scores = self.bm25.get_scores(q_tokens) if q_tokens else np.zeros(n)
        vec_scores = self.embeddings @ q_emb

        # 用排名做 RRF 融合，避免两类分数量纲不一致
        bm25_rank = np.argsort(bm25_scores)[::-1]
        vec_rank = np.argsort(vec_scores)[::-1]
        fused = self._rrf([bm25_rank, vec_rank])

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        results = []
        for idx, score in ordered:
            results.append(
                {
                    "chunk_id": int(idx),
                    "document": self.metadata[idx]["document"],
                    "text": self.chunks[idx],
                    "score": round(float(score), 4),
                }
            )
        return results

    @staticmethod
    def _rrf(rank_lists: List[np.ndarray], k: int = 60) -> dict:
        scores: dict = {}
        for ranks in rank_lists:
            for rank, idx in enumerate(ranks):
                scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
        return scores
