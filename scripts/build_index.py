"""离线构建检索索引：加载 data/docs 下的文档 → 切块 → 向量化 → 落盘。

用法：
    python scripts/build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.core.chunker import chunk_document
from app.core.embedder import Embedder
from app.core.loader import load_all
from app.core.retriever import HybridRetriever
from app.storage.index_store import IndexStore


def main() -> None:
    docs = load_all(Path(settings.docs_dir))
    if not docs:
        print(f"[警告] {settings.docs_dir} 下没有可索引的文档，请先放入 PDF/Word/Markdown 文件。")
        return

    chunks, metadata = [], []
    for name, text in docs:
        for chunk in chunk_document(text, settings.chunk_size, settings.chunk_overlap):
            chunks.append(chunk)
            metadata.append({"document": name})

    print("加载 Embedding 模型（首次运行会自动下载）...")
    embedder = Embedder(settings.embedding_model)
    retriever = HybridRetriever(embedder)
    retriever.build(chunks, metadata)
    IndexStore(settings.index_dir).save(retriever)

    print(f"索引构建完成：{len(docs)} 份文档 → {len(chunks)} 个文本块")
    print(f"索引已保存到 {settings.index_dir}")


if __name__ == "__main__":
    main()
