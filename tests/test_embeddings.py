import numpy as np

from docmind.embeddings import OllamaEmbedder, default_prefixes


def test_nomic_uses_task_prefixes():
    assert default_prefixes("nomic-embed-text") == ("search_query: ", "search_document: ")


def test_other_models_have_no_prefixes():
    assert default_prefixes("qwen3-embedding:0.6b") == ("", "")
    assert default_prefixes("bge-m3") == ("", "")


def test_embed_documents_and_query_apply_prefix():
    embedder = OllamaEmbedder(model="nomic-embed-text")
    captured: dict = {}

    def fake_embed(texts):
        captured["texts"] = texts
        return np.ones((len(texts), 3), dtype=np.float32)

    embedder.embed = fake_embed
    embedder.embed_documents(["doc one"])
    assert captured["texts"] == ["search_document: doc one"]
    embedder.embed_query("find me")
    assert captured["texts"] == ["search_query: find me"]


def test_explicit_prefixes_override_defaults():
    embedder = OllamaEmbedder(model="nomic-embed-text", query_prefix="", doc_prefix="")
    assert embedder.query_prefix == ""
    assert embedder.doc_prefix == ""
