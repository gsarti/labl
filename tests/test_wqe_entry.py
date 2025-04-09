import pytest

from wqe.data import WQEEntry


def test_from_edits_gap_merging():
    """Test that the from_edits method with gap merging is equivalent to merging post-initialization."""
    text = "This is a problematic sentence."
    edits = ["This is an edited good sentence.", "And this is a better edit."]
    entry_with_gaps = WQEEntry.from_edits(text=text, edits=edits, with_gaps=True)
    entry_without_gaps = WQEEntry.from_edits(text=text, edits=edits, with_gaps=False)
    with pytest.raises(RuntimeError):
        entry_without_gaps.merge_gap_annotations()
    entry_with_gaps.merge_gap_annotations()
    assert entry_with_gaps.tokens == entry_without_gaps.tokens
    assert entry_with_gaps.tokens_offsets == entry_without_gaps.tokens_offsets
    assert entry_with_gaps.edits_tokens == entry_without_gaps.edits_tokens
    assert entry_with_gaps.edits_tokens_offsets == entry_without_gaps.edits_tokens_offsets
