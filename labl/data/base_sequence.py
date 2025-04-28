from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt
from krippendorff.krippendorff import LevelOfMeasurement

from labl.data.base_entry import BaseLabeledEntry, EntryType
from labl.data.labeled_interface import LabeledInterface, LabeledObject
from labl.utils.agreement import AgreementOutput, get_labels_agreement
from labl.utils.token import LabeledToken, LabeledTokenList
from labl.utils.typing import LabelType

SequenceType = TypeVar("SequenceType", bound="BaseLabeledSequence")


class BaseLabeledSequence(LabeledInterface, list[LabeledObject], ABC):
    """Base class for all sequences classes containing `BaseLabeledEntry` objects.

    Supports basic list operations like indexing, length checking and iteration.
    """

    def __init__(self, iterable: Iterable[LabeledObject] | None = None):
        if iterable is None:
            super().__init__()
        else:
            super().__init__(iterable)
        self._label_types = self._get_label_types()

    def __sub__(self: SequenceType, other: SequenceType) -> SequenceType:
        return self.__class__(entry for entry in self if entry not in other)

    ### Utility Functions ###

    def relabel(
        self,
        relabel_fn: Callable[[LabelType], LabelType] | None = None,
        relabel_map: dict[str | int, LabelType] | None = None,
    ) -> None:
        """Relabels each entry in-place using a custom relabeling function or a mapping.

        Args:
            relabel_fn (Callable[[str | int | float | None], str | int | float | None]):
                A function that will be applied to each label in the entry.
                The function should take a single argument (the label) and return the new label.
                The function should return the label without any processing if the label should be preserved.
            relabel_map (dict[str | int, str | int | float | None]):
                A dictionary that maps old labels to new labels. The keys are the old labels and the values are the
                new labels. This can be used instead of the relabel_fn to relabel the entry if labels are discrete.
        """
        for entry in self:
            entry.relabel(relabel_fn=relabel_fn, relabel_map=relabel_map)

    ### Helper Functions ###

    def _get_label_types(self) -> list[type]:
        types = set()
        for entry in self:
            types.update(entry.label_types)
        return list(types)

    @abstractmethod
    def _get_labels_array(self, dtype: type | None = None) -> npt.NDArray[np.str_ | np.integer | np.floating]:
        pass


class BaseMultiLabelEntry(BaseLabeledSequence[EntryType], ABC):
    """Class for a list of `EntryType` objects representing multiple labels over the same text."""

    ### Getters and Setters ###

    @property
    def label_counts(self) -> list[int]:
        """Counts the number of labels for each token in the original text."""
        if not self:
            raise RuntimeError("No entries available.")
        per_token_labels: zip[tuple[LabelType, ...]] = zip(*[e.get_labels() for e in self], strict=True)
        return [sum([1 for lab in tok_labels if lab is not None]) for tok_labels in per_token_labels]

    @property
    def tokens_with_label_counts(self) -> LabeledTokenList:
        """Returns a list of `LabeledToken` with the number of labels for each token in the original text."""
        if not self:
            raise RuntimeError("No entries available.")
        return LabeledToken.from_list(list(zip(self[0].get_tokens(), self.label_counts, strict=True)))

    ### Utility Functions ###

    def get_agreement(
        self,
        level_of_measurement: LevelOfMeasurement | None = None,
    ) -> AgreementOutput:
        """Compute the inter-annotator agreement for the token labels of all label sets using
        [Krippendorff's alpha](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha).
        """
        self._validate_single_label_type()
        labels_array = self._get_labels_array(dtype=self.label_types[0])
        return get_labels_agreement(
            label_type=self.label_types[0],
            labels_array=labels_array,
            level_of_measurement=level_of_measurement,
        )

    ### Helper Functions ###

    def _get_labels_array(self, dtype: type | None = None) -> npt.NDArray[np.str_ | np.integer | np.floating]:
        if not self:
            raise RuntimeError("No entries available.")
        return self[0]._get_labels_array(self, dtype)


class BaseLabeledDataset(BaseLabeledSequence[EntryType | BaseMultiLabelEntry[EntryType]], ABC):
    """Base class for all dataset classes containing `BaseLabeledEntry` objects."""

    ### Utility Functions ###

    def get_agreement(
        self,
        other: BaseLabeledSequence[EntryType | BaseMultiLabelEntry[EntryType]] | None = None,
        level_of_measurement: LevelOfMeasurement | None = None,
    ) -> AgreementOutput:
        """Compute the inter-annotator agreement for the token labels of all label sets using
        [Krippendorff's alpha](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha).
        """
        self._validate_single_label_type()
        if other is not None:
            other._validate_single_label_type()
            if self.label_types[0] != other.label_types[0]:
                raise RuntimeError(
                    f"Label type does not match: {self.label_types[0]} vs {other.label_types[0]}.\n"
                    "Transform the annotations using `.relabel` to ensure a single type is present."
                )
        labels_array = self._get_labels_array(other=other, dtype=self.label_types[0])
        return get_labels_agreement(
            label_type=self.label_types[0],
            labels_array=labels_array,
            level_of_measurement=level_of_measurement,
        )

    ### Helper Functions ###

    def _get_labels_array(
        self, dtype: type | None = None, other=None
    ) -> npt.NDArray[np.str_ | np.integer | np.floating]:
        if not self:
            raise RuntimeError("No entries available.")
        is_multiedit_only = all(isinstance(entry, BaseMultiLabelEntry) for entry in self)
        is_single_edit_only = all(isinstance(entry, BaseLabeledEntry) for entry in self)
        if is_multiedit_only:
            return np.concat(
                [entry._get_labels_array(dtype) for entry in cast(list[BaseMultiLabelEntry], self)], axis=1
            )
        elif is_single_edit_only:
            if other is None:
                raise RuntimeError("`other` must be provided for datasets of single-label entries.")
            if not all(isinstance(entry, BaseLabeledEntry) for entry in other):
                raise RuntimeError("`other` must be a dataset of single-label entries.")
            if len(self) != len(other):
                raise RuntimeError(
                    f"Length of `self` ({len(self)}) and `other` ({len(other)}) must be the same to extract labels."
                )
            self_entries = cast(list[BaseLabeledEntry], self)
            other_entries = cast(list[BaseLabeledEntry], other)
            if not all(s.get_tokens() == o.get_tokens() for s, o in zip(self_entries, other_entries, strict=True)):
                raise RuntimeError("Tokens of all `self` and `other` entries must be the same to extract labels. ")
            return np.concatenate(
                [
                    self_entry._get_labels_array([self_entry, other_entry], dtype).astype(self.label_types[0])
                    for self_entry, other_entry in zip(self_entries, other_entries, strict=True)
                ],
                axis=1,
            )
        else:
            raise RuntimeError("All entries must have either a single or multiple labels to extract labels.")
