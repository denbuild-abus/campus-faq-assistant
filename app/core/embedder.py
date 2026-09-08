"""文本向量化：基于 sentence-transformers，模型惰性加载。"""
import numpy as np


class Embedder:
    """封装 Embedding 模型，首次 encode 时才真正加载（避免启动即下载模型）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts, normalize: bool = True) -> np.ndarray:
        """输入 str 或 list[str]，返回归一化后的 float32 向量矩阵。"""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        vecs = self.model.encode(
            texts, normalize_embeddings=normalize, show_progress_bar=False
        )
        return np.asarray(vecs, dtype="float32")
