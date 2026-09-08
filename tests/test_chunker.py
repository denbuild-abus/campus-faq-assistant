from app.core.chunker import chunk_document


def test_empty_text_returns_empty():
    assert chunk_document("") == []
    assert chunk_document("\n\n\n") == []


def test_short_text_is_single_chunk():
    chunks = chunk_document("新生报到需要带身份证。", chunk_size=256, overlap=32)
    assert len(chunks) == 1
    assert "身份证" in chunks[0]


def test_long_text_is_split_under_chunk_size():
    text = "\n".join(f"这是第 {i} 段内容，用于测试切块是否正常工作。" for i in range(200))
    chunks = chunk_document(text, chunk_size=128, overlap=16)
    assert len(chunks) > 1
    assert all(len(c) <= 128 for c in chunks)
