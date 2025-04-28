"""Aggregation functions to combine and summarize multiple labels.

Label aggregation is useful in the following scenarios:

1. Span -> Token Label Propagation (`LabeledEntry.get_tokens_from_spans`):
    A token might overlap with multiple spans, hence their labels should be aggregated over the token.
2. Gap merging for aligned sequences (`Tokenizer.merge_gap_annotations`):
    When special gap tokens that were inserted to hold insertion/deletions are merged to the right, their label should
    be combined with the label of the token to the right, if present.
2. Summarizing multi-edit entries:
    When multiple edits are available for the same text, an aggregation function can be used to summarize
    span and token labels.
"""

from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from labl.utils.typing import LabelType

T = TypeVar("T", bound=LabelType)


class LabelAggregation(Protocol):
    """Interface for label aggregation functions."""

    def __call__(self, labels: Sequence[LabelType]) -> Any:
        """Aggregate the labels.

        Args:
            labels (Sequence[Any]): The labels to aggregate.

        Returns:
            Any: The aggregated label.
        """
        ...


def label_sum_aggregation(labels: Sequence[T]) -> T | None:
    if not labels:
        return None
    out = labels[0]
    for l in labels[1:]:
        if type(l) is not type(out):
            raise RuntimeError(f"Different types found during aggregation: {type(l)} and {type(out)}")
        out += l  # type: ignore
    return out
