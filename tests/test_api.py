import numpy as np
from fastapi.testclient import TestClient

from app.core.pipeline import FAQPipeline
from app.core.retriever import HybridRetriever
from app.main import create_app


class FakeEmbedder:
    def embed(self, texts, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        return np.random.rand(len(texts), 8).astype("float32")


class FakeGenerator:
    model = "fake-model"

    def generate(self, query, contexts):
        return f"依据 {len(contexts)} 段资料回答：{query}"


def _make_client():
    retriever = HybridRetriever(FakeEmbedder())
    retriever.build(
        ["宿舍周日到周四 23:00 熄灯", "校园卡可在食堂自助机充值"],
        [{"document": "faq.md"}, {"document": "faq.md"}],
    )
    pipeline = FAQPipeline(retriever, FakeGenerator())
    return TestClient(create_app(pipeline))


def test_health():
    client = _make_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["indexed_chunks"] == 2


def test_chat_returns_answer_and_sources():
    client = _make_client()
    resp = client.post("/api/chat", json={"query": "宿舍几点熄灯"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"]
    assert len(data["sources"]) > 0
    assert data["model"] == "fake-model"


def test_chat_rejects_empty_query():
    client = _make_client()
    resp = client.post("/api/chat", json={"query": ""})
    assert resp.status_code == 422
