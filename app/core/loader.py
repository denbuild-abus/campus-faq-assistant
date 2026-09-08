"""文档加载：从 PDF / Word / 纯文本提取正文。"""
from pathlib import Path
from typing import List, Tuple

import docx
import pypdf


def load_document(path: Path) -> str:
    """按扩展名分发，返回文档纯文本。"""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext in (".docx", ".doc"):
        return _load_docx(path)
    if ext in (".txt", ".md", ".markdown", ".rst"):
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"不支持的文件类型: {ext}（支持 PDF / Word / 纯文本）")


def _load_pdf(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(parts)


def _load_docx(path: Path) -> str:
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def load_all(docs_dir: Path) -> List[Tuple[str, str]]:
    """加载目录下所有受支持文档，返回 (文件名, 文本) 列表。"""
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return []
    supported = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".rst"}
    results: List[Tuple[str, str]] = []
    for path in sorted(docs_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in supported:
            try:
                results.append((path.name, load_document(path)))
            except Exception as exc:  # 单个文件失败不阻断整体
                print(f"[警告] 跳过 {path.name}: {exc}")
    return results
