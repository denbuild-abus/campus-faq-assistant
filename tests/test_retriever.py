import numpy as np

from app.core.retriever import HybridRetriever, tokenize


class FakeEmbedder:
    """不依赖模型、确定性输出的假向量器，供单元测试使用。"""

    def __init__(self, dim=8):
        self.dim = dim

    def embed(self, texts, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            seed = sum(ord(c) for c in t) % 100
            rng = np.random.default_rng(seed)
            v = rng.random(self.dim).astype("float32")
            v = v / (np.linalg.norm(v) + 1e-8)
            vecs.append(v)
        return np.array(vecs, dtype="float32")


def test_tokenize_chinese_and_english():
    tokens = tokenize("校园卡如何充值 支持 visa card")
    assert tokens  # 至少能切出词元
    assert "visa" in tokens or "card" in tokens


def test_retriever_build_and_search():
    embedder = FakeEmbedder()
    retriever = HybridRetriever(embedder)
    chunks = ["宿舍周日到周四 23:00 熄灯", "校园卡可在食堂自助机充值", "选课在开学第二周开始"]
    meta = [{"document": "faq.md"} for _ in chunks]
    retriever.build(chunks, meta)

    results = retriever.search("宿舍几点熄灯", top_k=2)
    assert results
    assert results[0]["document"] == "faq.md"
    assert "熄灯" in results[0]["text"]


def test_retriever_empty_returns_nothing():
    retriever = HybridRetriever(FakeEmbedder())
    assert retriever.search("任何问题") == []
