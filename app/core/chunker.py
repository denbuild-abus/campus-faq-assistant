"""文本切块：按段落聚合，达到目标长度则切分，相邻块保留重叠。"""
from typing import List


def chunk_document(text: str, chunk_size: int = 256, overlap: int = 32) -> List[str]:
    """把长文本切成若干文本块。

    策略：先按空行/换行拆成段落，再贪心地把段落合并进当前块；
    单段超过 chunk_size 时硬切，并在块与块之间保留 overlap 个字符，
    避免关键信息被截断。
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 1 <= chunk_size:
            current = current + "\n" + para
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = (tail + "\n" + para).strip() if tail else para

        # 单段超长时硬切
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - overlap:]

    if current:
        chunks.append(current)
    return chunks
