"""A small evaluation harness for comparing models and retrieval settings."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from docmind.rag import RAGEngine

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).lower().replace(",", "")


@dataclass
class EvalItem:
    question: str
    keywords: list[str]
    source: str | None = None


@dataclass
class ItemResult:
    question: str
    source: str | None
    retrieval_hit: bool
    source_hit: bool
    answer_hit: bool | None
    matched: str
    retrieved_sources: list[str] = field(default_factory=list)


def load_golden(path: str | Path) -> list[EvalItem]:
    """Load a JSONL golden set of ``{question, keywords, source}`` items."""
    items: list[EvalItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        items.append(
            EvalItem(
                question=data["question"],
                keywords=[str(k) for k in data.get("keywords", [])],
                source=data.get("source"),
            )
        )
    return items


def _first_match(text: str, keywords: list[str]) -> str:
    haystack = _normalize(text)
    for keyword in keywords:
        if _normalize(keyword) in haystack:
            return keyword
    return ""


def evaluate(engine: RAGEngine, items: list[EvalItem], generate: bool = True) -> list[ItemResult]:
    """Run each golden question through the engine and score the outcome."""
    results: list[ItemResult] = []
    for item in items:
        sources = engine.retrieve(item.question)
        context = "\n".join(source.text for source in sources)
        retrieved_names = [source.source for source in sources]

        retrieval_hit = bool(_first_match(context, item.keywords))
        matched = _first_match(context, item.keywords)
        source_hit = True if not item.source else any(item.source in name for name in retrieved_names)

        answer_hit: bool | None = None
        if generate:
            answer = engine.query(item.question)
            matched = _first_match(answer.text, item.keywords) or matched
            answer_hit = bool(_first_match(answer.text, item.keywords))

        results.append(
            ItemResult(
                question=item.question,
                source=item.source,
                retrieval_hit=retrieval_hit,
                source_hit=source_hit,
                answer_hit=answer_hit,
                matched=matched,
                retrieved_sources=retrieved_names,
            )
        )
    return results


def summarize(results: list[ItemResult]) -> dict:
    """Aggregate hit rates across all evaluated items."""
    total = len(results) or 1
    answered = [r for r in results if r.answer_hit is not None]
    return {
        "items": len(results),
        "retrieval_hit_rate": sum(r.retrieval_hit for r in results) / total,
        "source_hit_rate": sum(r.source_hit for r in results) / total,
        "answer_hit_rate": (sum(bool(r.answer_hit) for r in answered) / len(answered)) if answered else None,
    }


def format_report(results: list[ItemResult], model: str, options: dict) -> str:
    """Render a human-readable comparison table."""
    summary = summarize(results)
    lines = [
        f"Model: {model}",
        (
            f"Options: temperature={options.get('temperature')} top_p={options.get('top_p')} "
            f"num_predict={options.get('num_predict')}"
        ),
        "",
        f"{'#':>2}  {'retr':<4} {'src':<4} {'ans':<4} {'match':<12} question",
        "-" * 78,
    ]
    for i, result in enumerate(results, start=1):
        ans = "-" if result.answer_hit is None else ("yes" if result.answer_hit else "no")
        lines.append(
            f"{i:>2}  {'yes' if result.retrieval_hit else 'no':<4} "
            f"{'yes' if result.source_hit else 'no':<4} {ans:<4} "
            f"{result.matched[:12]:<12} {result.question}"
        )
    lines += [
        "-" * 78,
        f"retrieval hit@k : {summary['retrieval_hit_rate']:.0%}",
        f"expected source : {summary['source_hit_rate']:.0%}",
    ]
    if summary["answer_hit_rate"] is not None:
        lines.append(f"answer hit rate : {summary['answer_hit_rate']:.0%}")
    return "\n".join(lines)
