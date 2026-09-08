"""检索效果评估：计算 Hit@k 与 MRR@k。

测试集格式（data/eval_questions.json）：
    [{"query": "问题", "expected_document": "文件名"}]

用法：
    python scripts/eval_rag.py [--top-k 5]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.core.embedder import Embedder
from app.core.retriever import HybridRetriever
from app.storage.index_store import IndexStore


def evaluate(retriever: HybridRetriever, test_set: list, top_k: int) -> dict:
    hits = 0
    rr_sum = 0.0
    for item in test_set:
        query = item["query"]
        expected = item["expected_document"]
        results = retriever.search(query, top_k=top_k)
        docs = [r["document"] for r in results]
        if expected in docs:
            hits += 1
            rr_sum += 1.0 / (docs.index(expected) + 1)
    n = len(test_set) or 1
    return {"hit": hits / n, "mrr": rr_sum / n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    test_path = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"
    if not test_path.exists():
        print(f"[错误] 找不到测试集 {test_path}")
        sys.exit(1)
    test_set = json.loads(test_path.read_text(encoding="utf-8"))

    store = IndexStore(settings.index_dir)
    if not store.exists():
        print("[错误] 索引不存在，请先运行 scripts/build_index.py")
        sys.exit(1)

    embedder = Embedder(settings.embedding_model)
    retriever = store.load(HybridRetriever(embedder))
    metrics = evaluate(retriever, test_set, top_k=args.top_k)

    print(f"测试集规模：{len(test_set)} 题")
    print(f"Hit@{args.top_k}：{metrics['hit']:.2%}")
    print(f"MRR@{args.top_k}：{metrics['mrr']:.4f}")


if __name__ == "__main__":
    main()
