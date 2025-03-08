import pytest

from wqeval import WordLevelQEEntry


def test_basic_tagging():
    orig = "This is a <h>highlighted</h> text."
    entry = WordLevelQEEntry.from_tagged_text(orig)
    assert entry.orig == "This is a highlighted text."
    assert entry.tagged_spans == [{"start": 10, "end": 21, "tag": "h", "text": "highlighted"}]


def test_multiple_tags_same_type():
    orig = "This <h>is</h> a <h>highlighted</h> text."
    entry = WordLevelQEEntry.from_tagged_text(orig)
    assert entry.orig == "This is a highlighted text."
    assert entry.tagged_spans == [
        {"start": 5, "end": 7, "tag": "h", "text": "is"},
        {"start": 10, "end": 21, "tag": "h", "text": "highlighted"},
    ]


def test_multiple_tags_different_types():
    orig = "This <h>is</h> a <error>highlighted</error> text."
    entry = WordLevelQEEntry.from_tagged_text(orig, keep_tag_types=["h", "error"])
    assert entry.orig == "This is a highlighted text."
    assert entry.tagged_spans == [
        {"start": 5, "end": 7, "tag": "h", "text": "is"},
        {"start": 10, "end": 21, "tag": "error", "text": "highlighted"},
    ]


def test_ignored_tag_types():
    orig = "This <h>is</h> a <error>highlighted</error> text."
    entry = WordLevelQEEntry.from_tagged_text(orig, keep_tag_types=["h"], ignore_tag_types=["error"])
    assert entry.orig == "This is a highlighted text."
    assert entry.tagged_spans == [{"start": 5, "end": 7, "tag": "h", "text": "is"}]


def test_consecutive_tags():
    """Test consecutive highlights without spaces."""
    orig = "This <h>is</h><error>highlighted</error> text."
    entry = WordLevelQEEntry.from_tagged_text(orig, keep_tag_types=["h", "error"])
    assert entry.orig == "This ishighlighted text."
    assert entry.tagged_spans == [
        {"start": 5, "end": 7, "tag": "h", "text": "is"},
        {"start": 7, "end": 18, "tag": "error", "text": "highlighted"},
    ]


def test_nested_tags_error():
    orig = "This <h>is <error>highlighted</h> text</error>."
    with pytest.raises(RuntimeError, match="Closing tag"):
        WordLevelQEEntry.from_tagged_text(orig, keep_tag_types=["h", "error"])


def test_unclosed_tag_error():
    orig = "This <h>is highlighted text."
    with pytest.raises(RuntimeError, match="Unclosed tags"):
        WordLevelQEEntry.from_tagged_text(orig)


def test_unopened_tag_error():
    orig = "This is highlighted</h> text."
    with pytest.raises(RuntimeError, match="Closing tag"):
        WordLevelQEEntry.from_tagged_text(orig)


def test_empty_input_error():
    """Test that empty input raises an error."""
    with pytest.raises(RuntimeError, match="Text cannot be empty"):
        WordLevelQEEntry.from_tagged_text("")


def test_empty_tag_types_error():
    with pytest.raises(RuntimeError, match="At least one tag type must be provided in keep_tag_types"):
        WordLevelQEEntry.from_tagged_text("Some text", keep_tag_types=[])


def test_multiple_tags_with_whitespace():
    orig = "This <h>is</h> a <h>  highlighted  </h> text."
    entry = WordLevelQEEntry.from_tagged_text(orig)
    assert entry.orig == "This is a   highlighted   text."
    assert entry.tagged_spans == [
        {"start": 5, "end": 7, "tag": "h", "text": "is"},
        {"start": 10, "end": 25, "tag": "h", "text": "  highlighted  "},
    ]


def test_no_tagged_sections():
    orig = "This is a plain text without highlights."
    entry = WordLevelQEEntry.from_tagged_text(orig)
    assert entry.orig == orig
    assert entry.tagged_spans == []


def test_tag_at_beginning():
    orig = "<h>This</h> is a text."
    entry = WordLevelQEEntry.from_tagged_text(orig)
    assert entry.orig == "This is a text."
    assert entry.tagged_spans == [{"start": 0, "end": 4, "tag": "h", "text": "This"}]


def test_tag_at_end():
    orig = "This is a <h>text.</h>"
    entry = WordLevelQEEntry.from_tagged_text(orig)
    assert entry.orig == "This is a text."
    assert entry.tagged_spans == [{"start": 10, "end": 15, "tag": "h", "text": "text."}]


def test_entire_text_highlighted():
    orig = "<h>This is all highlighted text.</h>"
    entry = WordLevelQEEntry.from_tagged_text(orig)
    assert entry.orig == "This is all highlighted text."
    assert entry.tagged_spans == [{"start": 0, "end": 29, "tag": "h", "text": "This is all highlighted text."}]


def test_unexpected_tag_type():
    orig = "This <h>is</h> a <unexpected>highlighted</unexpected> text."
    with pytest.warns(
        UserWarning, match="The text contains tag types that were not specified in keep_tag_types or ignore_tag_types"
    ):
        entry = WordLevelQEEntry.from_tagged_text(orig)
    assert entry.orig == "This is a <unexpected>highlighted</unexpected> text."
