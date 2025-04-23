import pytest

from wqe.data import EditedEntry, MultiEditedEntry


def test_from_edits_gap_merging():
    """Test that the from_edits method with gap merging is equivalent to merging post-initialization."""
    text = "This is a problematic sentence."
    edits = ["This is an edited good sentence.", "And this is a better edit."]
    entry_with_gaps = EditedEntry.from_edits(text=text, edits=edits, with_gaps=True)
    entry_without_gaps = EditedEntry.from_edits(text=text, edits=edits, with_gaps=False)
    assert isinstance(entry_with_gaps, MultiEditedEntry)
    assert isinstance(entry_without_gaps, MultiEditedEntry)
    with pytest.raises(RuntimeError):
        entry_without_gaps.merge_gap_annotations()
    entry_with_gaps.merge_gap_annotations()
    assert entry_with_gaps[0]._orig._tokens == entry_without_gaps[0]._orig._tokens
    assert entry_with_gaps[0]._orig._tokens_offsets == entry_without_gaps[0]._orig._tokens_offsets
    assert entry_with_gaps[0]._orig._tokens_labels == entry_without_gaps[0]._orig._tokens_labels
    assert entry_with_gaps[0]._edit._tokens == entry_without_gaps[0]._edit._tokens
    assert entry_with_gaps[0]._edit._tokens_offsets == entry_without_gaps[0]._edit._tokens_offsets
    assert entry_with_gaps[0]._edit._tokens_labels == entry_without_gaps[0]._edit._tokens_labels
