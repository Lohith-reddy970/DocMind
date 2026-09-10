import json
from dataclasses import dataclass

from docmind.config import Settings
from docmind.eval import EvalItem, ItemResult, evaluate, load_golden, summarize


@dataclass
class _Source:
    text: str
    source: str


@dataclass
class _Answer:
    text: str


class _StubEngine:
    def __init__(self, answer: str = "the answer is 20 days", source: str = "handbook.md"):
        self.answer = answer
        self.source = source

    def retrieve(self, question):
        return [_Source(self.answer, self.source)]

    def query(self, question):
        return _Answer(self.answer)


def _items(rows):
    return [EvalItem(r["question"], r["keywords"], r.get("source")) for r in rows]


def test_load_golden(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(json.dumps({"question": "Q", "keywords": ["a"], "source": "s.txt"}), encoding="utf-8")
    items = load_golden(path)
    assert items[0].question == "Q"
    assert items[0].keywords == ["a"]


def test_evaluate_scores_hits():
    items = _items([{"question": "vacation", "keywords": ["20 days"], "source": "handbook.md"}])
    results = evaluate(_StubEngine(), items, generate=True)
    assert results[0].retrieval_hit
    assert results[0].source_hit
    assert results[0].answer_hit


def test_evaluate_detects_miss():
    items = _items([{"question": "rockets", "keywords": ["thrust"], "source": "nasa.md"}])
    results = evaluate(_StubEngine(), items, generate=False)
    assert not results[0].retrieval_hit
    assert not results[0].source_hit
    assert results[0].answer_hit is None


def test_summarize_rates():
    results = [
        ItemResult("q", "s", True, True, True, "20 days"),
        ItemResult("q", "s", False, False, False, ""),
    ]
    summary = summarize(results)
    assert summary["retrieval_hit_rate"] == 0.5
    assert summary["answer_hit_rate"] == 0.5


def test_llm_options_exposed():
    opts = Settings().llm_options()
    assert {"temperature", "top_p", "repeat_penalty", "num_predict", "num_ctx"} <= set(opts)
