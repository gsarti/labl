from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

from labl.utils.typing import InfoDictType, LabelType, SerializableDictType

LabeledObject = TypeVar("LabeledObject", bound="LabeledInterface")
SerializedLabeledObject = TypeVar("SerializedLabeledObject", bound="SerializableDictType")


class LabeledInterface(ABC):
    _label_types: list[type]
    _info: InfoDictType

    ### Getters and Setters ###

    @property
    def info(self) -> InfoDictType:
        """Returns the info dictionary for the entry. This is a read-only property."""
        return self._info

    @info.setter
    def info(self, info: InfoDictType):
        raise RuntimeError("Cannot set the attribute `info` after initialization")

    @property
    def label_types(self) -> list[type]:
        """Returns the list of label types for the entry. This is a read-only property."""
        return self._label_types

    @label_types.setter
    def label_types(self, t: list[type]):
        raise RuntimeError("Cannot set the attribute `label_types` after initialization")

    ### Utility Functions ###

    @abstractmethod
    def relabel(
        self,
        relabel_fn: Callable[[LabelType], LabelType] | None = None,
        relabel_map: dict[str | int, LabelType] | None = None,
    ) -> None:
        pass

    @abstractmethod
    def to_dict(self) -> SerializableDictType:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: SerializedLabeledObject) -> "LabeledInterface":  # type: ignore
        """Creates an instance of the class from a dictionary representation."""
        pass

    ### Helper Functions ###

    @abstractmethod
    def _get_label_types(self) -> list[type]:
        pass

    def _validate_single_label_type(self) -> None:
        if len(self.label_types) > 1:
            raise RuntimeError(
                f"Multiple label types found: {','.join(str(t) for t in self.label_types)}.\n"
                "Transform the annotations using `.relabel` to ensure a single type is present."
            )
        elif len(self.label_types) == 0:
            raise RuntimeError("No labels found for entries.")
