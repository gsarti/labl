from typing import cast

import pytest

from labl.data import EditedEntry, MultiEditEntry


def test_from_edits_gap_merging():
    """Test that the from_edits method with gap merging is equivalent to merging post-initialization."""
    text = "This is a problematic sentence."
    edits = ["This is an edited good sentence.", "And this is a better edit."]
    entry_with_gaps = EditedEntry.from_edits(text=text, edits=edits, with_gaps=True)
    entry_without_gaps = EditedEntry.from_edits(text=text, edits=edits, with_gaps=False)
    assert isinstance(entry_with_gaps, MultiEditEntry)
    assert isinstance(entry_without_gaps, MultiEditEntry)
    with pytest.raises(RuntimeError):
        entry_without_gaps.merge_gap_annotations()
    entry_with_gaps.merge_gap_annotations()
    assert entry_with_gaps[0].orig.tokens == entry_without_gaps[0].orig.tokens
    assert entry_with_gaps[0].orig.tokens_offsets == entry_without_gaps[0].orig.tokens_offsets
    assert entry_with_gaps[0].orig.tokens_labels == entry_without_gaps[0].orig.tokens_labels
    assert entry_with_gaps[0].edit.tokens == entry_without_gaps[0].edit.tokens
    assert entry_with_gaps[0].edit.tokens_offsets == entry_without_gaps[0].edit.tokens_offsets
    assert entry_with_gaps[0].edit.tokens_labels == entry_without_gaps[0].edit.tokens_labels


def test_relabel():
    """Test that relabeling with edit counts works correctly."""
    text = "This is a problematic sentence."
    edit = "This is an edited good sentence."
    entry = EditedEntry.from_edits(text=text, edits=edit, with_gaps=True)
    entry = cast(EditedEntry, entry)
    assert "I" in entry.edit.tokens_labels
    entry.relabel(lambda x: x if x != "I" else None)
    assert "I" not in entry.edit.tokens_labels
