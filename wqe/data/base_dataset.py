from collections.abc import Callable
from typing import Generic, TypeVar

from wqe.data.base_entry import BaseEntry
from wqe.utils.typing import LabelType

EntryType = TypeVar("EntryType", bound=BaseEntry)
DatasetType = TypeVar("DatasetType", bound="BaseDataset")


class BaseDataset(Generic[EntryType]):
    """Base class for all dataset classes containing `BaseEntry` objects.

    Attributes:
        data (list[LabeledEntry]): A list of `BaseEntry` objects.
    """

    def __init__(self, data: list[EntryType]):
        """Initialize the dataset with a list of entry objects.

        Args:
            data (list): A list of LabeledEntry objects.
        """
        self.data: list[EntryType] = data
        self._label_types: list[type] = self._get_label_types()

    def __getitem__(self, idx) -> EntryType:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __add__(self: DatasetType, other: DatasetType) -> DatasetType:
        return self.__class__(self.data + other.data)

    def __sub__(self: DatasetType, other: DatasetType) -> DatasetType:
        return self.__class__([entry for entry in self.data if entry not in other.data])

    ### Getters and Setters ###

    @property
    def label_types(self) -> list[type]:
        return self._label_types

    @label_types.setter
    def label_types(self, t: list[type]):
        raise RuntimeError("Cannot set the attribute `label_types` after initialization")

    ### Helper Functions ###

    def _get_label_types(self) -> list[type]:
        all_types = set()
        for entry in self.data:
            all_types.union(set(entry._label_types))
        return list(all_types)

    ### Utility Functions ###

    def relabel(
        self,
        relabel_fn: Callable[[LabelType], LabelType] | None = None,
        relabel_map: dict[str | int, LabelType] | None = None,
    ) -> None:
        """Relabels each dataset entry in-place using a custom relabeling function or a mapping.

        Args:
            relabel_fn (Callable[[str | int | float | None], str | int | float | None]):
                A function that will be applied to each label in the entry.
                The function should take a single argument (the label) and return the new label.
                The function should return the label without any processing if the label should be preserved.
            relabel_map (dict[str | int, str | int | float | None]):
                A dictionary that maps old labels to new labels. The keys are the old labels and the values are the
                new labels. This can be used instead of the relabel_fn to relabel the entry if labels are discrete.
        """
        for entry in self.data:
            entry.relabel(relabel_fn=relabel_fn, relabel_map=relabel_map)
