from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt
from krippendorff.krippendorff import LevelOfMeasurement

from labl.data.labeled_interface import LabeledInterface
from labl.utils.agreement import AgreementOutput, get_labels_agreement
from labl.utils.typing import LabelType

EntryType = TypeVar("EntryType", bound="BaseLabeledEntry")


class BaseLabeledEntry(LabeledInterface, ABC):
    """Base class for all data entries. This class handles the creation of public getters, disallowing setting and
    providing a private constructor key to prevent direct instantiation.
    """

    ### Utility Functions ###

    @abstractmethod
    def get_tokens(self) -> list[str]:
        pass

    @abstractmethod
    def get_labels(self) -> Sequence[LabelType]:
        pass

    def relabel(
        self,
        relabel_fn: Callable[[LabelType], LabelType] | None = None,
        relabel_map: dict[str | int, LabelType] | None = None,
    ) -> None:
        """Relabels the entry in-place using a custom relabeling function or a mapping.

        Args:
            relabel_fn (Callable[[str | int | float | None], str | int | float | None]):
                A function that will be applied to each label in the entry.
                The function should take a single argument (the label) and return the new label.
                The function should return the label without any processing if the label should be preserved.
            relabel_map (dict[str | int, str | int | float | None]):
                A dictionary that maps old labels to new labels. The keys are the old labels and the values are the
                new labels. This can be used instead of the relabel_fn to relabel the entry if labels are discrete.
        """
        if relabel_fn is None:
            if relabel_map is None:
                raise ValueError("Either relabel_fn or relabel_map must be provided.")
            relabel_fn = lambda x: x if x is None or isinstance(x, float) else relabel_map.get(x, x)
        self._relabel_attributes(relabel_fn)
        self._label_types = self._get_label_types()

    def get_agreement(
        self: EntryType,
        other: EntryType,
        level_of_measurement: LevelOfMeasurement | None = None,
    ) -> AgreementOutput:
        """Compute the inter-annotator agreement for the token labels of two entries using
        [Krippendorff's alpha](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha).
        """
        self._validate_single_label_type()
        other._validate_single_label_type()
        if self.label_types[0] != other.label_types[0]:
            raise RuntimeError(
                f"Label type does not match: {self.label_types[0]} vs {other.label_types[0]}.\n"
                "Transform the annotations using `.relabel` to ensure a single type is present."
            )
        labels_array = self._get_labels_array([self, other], dtype=self.label_types[0])
        return get_labels_agreement(
            label_type=self.label_types[0],
            labels_array=labels_array,
            level_of_measurement=level_of_measurement,
        )

    ### Helper Functions ###

    @abstractmethod
    def _relabel_attributes(self, relabel_fn: Callable[[LabelType], LabelType]) -> None:
        pass

    @abstractmethod
    def _get_labels_array(
        self, items: Sequence[EntryType], dtype: type | None = None
    ) -> npt.NDArray[np.str_ | np.integer | np.floating]:
        pass
