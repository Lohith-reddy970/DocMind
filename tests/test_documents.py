import pytest

from docmind.documents import load_file, load_path


def test_load_file_strips_surrounding_quotes(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("hello world", encoding="utf-8")
    doc = load_file(f'"{target}"')
    assert "hello world" in doc.text


def test_load_path_handles_quoted_directory(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    docs = load_path(f"'{tmp_path}'")
    assert len(docs) == 2


def test_unsupported_extension_raises(tmp_path):
    target = tmp_path / "data.csv"
    target.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError):
        load_file(target)


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_path("does/not/exist.txt")
