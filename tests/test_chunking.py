from docmind.chunking import chunk_document, split_text
from docmind.documents import Document


def test_split_respects_size():
    text = "\n\n".join("word " * 40 for _ in range(6))
    chunks = split_text(text, chunk_size=200, overlap=40)
    assert chunks
    assert all(len(c) <= 260 for c in chunks)


def test_split_overlap_present():
    text = ("alpha " * 100).strip()
    chunks = split_text(text, chunk_size=120, overlap=30)
    assert len(chunks) > 1


def test_empty_text_returns_no_chunks():
    assert split_text("   ", chunk_size=100, overlap=10) == []


def test_chunk_document_indexes_sequentially():
    doc = Document(text="sentence. " * 200, source="a.txt", metadata={"type": "txt"})
    chunks = chunk_document(doc, chunk_size=150, overlap=30)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.source == "a.txt" for c in chunks)


def test_chunk_ids_unique():
    doc = Document(text="data " * 300, source="b.md", metadata={})
    chunks = chunk_document(doc, chunk_size=150, overlap=20)
    assert len({c.id for c in chunks}) == len(chunks)
