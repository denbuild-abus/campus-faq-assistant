"""问答编排：检索 → 生成，含大模型调用失败兜底。"""
from app.core.generator import Generator
from app.core.retriever import HybridRetriever


class FAQPipeline:
    def __init__(self, retriever: HybridRetriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def ask(self, query: str, top_k: int = 5) -> dict:
        """返回 {"answer": str, "sources": list}。"""
        if not self.retriever.chunks:
            return {
                "answer": "知识库为空，请先上传文档（POST /api/documents）或运行 scripts/build_index.py 构建索引。",
                "sources": [],
            }
        sources = self.retriever.search(query, top_k=top_k)
        try:
            answer = self.generator.generate(query, sources)
        except Exception as exc:
            # 兜底：大模型不可用时返回最相关原文，保证服务不崩
            answer = sources[0]["text"] + f"\n\n（大模型调用失败：{exc}，已返回最相关原文）"
        return {"answer": answer, "sources": sources}
