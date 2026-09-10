import numpy as np
import pytest

from docmind.chunking import Chunk
from docmind.vector_store import VectorStore


def make_chunk(index: int) -> Chunk:
    return Chunk(text=f"chunk {index}", source="doc.txt", chunk_index=index, metadata={})


def test_add_and_search_orders_by_similarity():
    store = VectorStore()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    store.add(vectors, [make_chunk(i) for i in range(3)])

    results = store.search(np.array([1.0, 0.0]), top_k=2)
    assert results[0][0].chunk_index == 0
    assert results[0][1] > results[1][1]


def test_dimension_mismatch_raises():
    store = VectorStore()
    store.add(np.ones((1, 3), dtype=np.float32), [make_chunk(0)])
    with pytest.raises(ValueError):
        store.add(np.ones((1, 4), dtype=np.float32), [make_chunk(1)])


def test_save_and_load_roundtrip(tmp_path):
    store = VectorStore()
    store.add(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), [make_chunk(0)])
    store.save(tmp_path / "v.npy", tmp_path / "c.jsonl")

    loaded = VectorStore.load(tmp_path / "v.npy", tmp_path / "c.jsonl")
    assert loaded.size == 1
    assert loaded.dim == 3
    assert loaded.search(np.array([1.0, 2.0, 3.0]), top_k=1)[0][0].text == "chunk 0"


def test_load_missing_files_is_empty(tmp_path):
    loaded = VectorStore.load(tmp_path / "none.npy", tmp_path / "none.jsonl")
    assert loaded.size == 0
