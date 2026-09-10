from docmind.llm import _dedupe_stream, build_prompt, dedupe_repeated_sentences


def test_dedupe_collapses_immediate_repeat():
    text = "I don't know. I don't know."
    assert dedupe_repeated_sentences(text) == "I don't know."


def test_dedupe_keeps_distinct_sentences():
    text = "First fact. Second fact."
    assert dedupe_repeated_sentences(text) == text


def test_stream_dedupes_back_to_back_sentences():
    tokens = ["Answer one. ", "Answer one. ", "Answer two. "]
    assert "".join(_dedupe_stream(iter(tokens))) == "Answer one. Answer two. "


def test_stream_preserves_unique_text():
    tokens = ["Hello ", "world. ", "How ", "are you? "]
    assert "".join(_dedupe_stream(iter(tokens))) == "Hello world. How are you? "


def test_build_prompt_includes_sources_and_question():
    prompt = build_prompt("What?", [("a.txt", "the answer is 42")])
    assert "a.txt" in prompt
    assert "the answer is 42" in prompt
    assert "What?" in prompt


def test_build_prompt_handles_no_context():
    assert "no relevant context" in build_prompt("What?", [])
