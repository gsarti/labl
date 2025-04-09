"""Aggregation functions to combine and summarize multiple labels.

Label aggregation is useful in the following scenarios:

1. Span -> Token Label Propagation (`WQEEntry.get_tokens_from_spans`):
    A token might overlap with multiple spans, hence their labels should be aggregated over the token.
2. Gap merging for aligned sequences (`WQEEntry.merge_gap_annotations`):
    When special gap tokens that were inserted to hold insertion/deletions are merged to the right, their label should
    be combined with the label of the token to the right, if present.
2. Summarizing multi-edit entries:
    When multiple edits are available for the same text, an aggregation function can be used to summarize
    span and token labels.
"""

from collections.abc import Sequence
from typing import Any, Protocol


class LabelAggregation(Protocol):
    """Interface for label aggregation functions."""

    def __call__(self, labels: Sequence[str | int | float | None]) -> Any:
        """Aggregate the labels.

        Args:
            labels (Sequence[Any]): The labels to aggregate.

        Returns:
            Any: The aggregated label.
        """
        ...


def label_count_aggregation(labels: Sequence[str | int | float | None]) -> int:
    """Aggregation function summarizing a set of labels by counting non-empty labels."""
    return len([l for l in labels if l is not None])


def label_sumlen_aggregation(labels: Sequence[str | int | float | None]) -> int:
    """Aggregation function summarizing a set of labels by summing their lengths."""
    return sum(len(str(l)) for l in labels if l is not None)
