from docmind.config import Settings
from docmind.rag import RAGEngine
from docmind.vector_store import VectorStore


def build_engine(settings, embedder, llm):
    return RAGEngine(settings=settings, embedder=embedder, llm=llm, store=VectorStore(), autoload=False)


def test_ingest_text_indexes_chunks(settings, fake_embedder, fake_llm):
    engine = build_engine(settings, fake_embedder, fake_llm)
    count = engine.ingest_text("The office is closed on weekends. " * 30, source="policy")
    assert count > 0
    assert engine.store.size == count


def test_query_returns_answer_and_sources(settings, fake_embedder, fake_llm):
    engine = build_engine(settings, fake_embedder, fake_llm)
    engine.ingest_text("Vacation policy grants twenty days per year. " * 20, source="handbook")
    answer = engine.query("How many vacation days?")
    assert answer.text == "test answer"
    assert answer.sources
    assert answer.sources[0].source == "handbook"


def test_empty_store_returns_no_sources(settings, fake_embedder, fake_llm):
    engine = build_engine(settings, fake_embedder, fake_llm)
    assert engine.retrieve("anything") == []
    answer = engine.query("anything")
    assert answer.sources == []


def test_persistence_roundtrip(settings, fake_embedder, fake_llm):
    engine = build_engine(settings, fake_embedder, fake_llm)
    engine.ingest_text("Persisted content. " * 20, source="doc")
    engine.save()

    reloaded = RAGEngine(settings=settings, embedder=fake_embedder, llm=fake_llm, autoload=True)
    assert reloaded.store.size == engine.store.size


def test_min_score_filters_weak_matches(settings, fake_embedder, fake_llm):
    strict = Settings(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        min_score=0.99,
        index_dir=settings.index_dir,
    )
    engine = build_engine(strict, fake_embedder, fake_llm)
    engine.ingest_text("Vacation policy grants twenty days per year. " * 20, source="handbook")
    assert engine.retrieve("unrelated question about rockets") == []
    assert engine.retrieve("Vacation policy grants twenty days per year.")


def test_stats_reports_counts(settings, fake_embedder, fake_llm):
    engine = build_engine(settings, fake_embedder, fake_llm)
    engine.ingest_text("Some content. " * 20, source="a")
    engine.ingest_text("More content. " * 20, source="b")
    stats = engine.stats()
    assert stats["sources"] == 2
    assert stats["chunks"] == engine.store.size
