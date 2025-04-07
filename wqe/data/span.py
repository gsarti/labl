from collections.abc import Sequence
from dataclasses import dataclass
from pprint import pformat
from typing import Generic, TypeVar

DictSpan = dict[str, str | int | float | None]
QESpanInput = Sequence["Span"] | Sequence[DictSpan]
QESpanWithEditInput = Sequence["EditSpan"] | Sequence[tuple[DictSpan, DictSpan]]

SpanType = TypeVar("SpanType", bound="BaseSpan")
LoadFromType = TypeVar("LoadFromType", bound=DictSpan | tuple[DictSpan, DictSpan])


class BaseSpan(Generic[LoadFromType]):
    """Base class for spans.
    This class provides a common interface for spans and allows for easy conversion
    between different representations (e.g., dict, tuple).

    Attributes:
        _load_from_type (type): The type of the data that can be loaded into this span using the `load` method.
    """

    _load_from_type: type = dict

    def to_dict(self) -> dict:
        """Converts the span to a dictionary."""
        return {
            k: v if not isinstance(v, BaseSpan) else v.to_dict()
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    @classmethod
    def load(cls: type[SpanType], data: LoadFromType) -> SpanType:
        """Creates a span instance from a primitive type."""
        if not isinstance(data, cls._load_from_type):
            raise TypeError(f"Invalid input type. Expected {str(cls._load_from_type)}, got {type(data)}.")
        return cls(**data)

    @classmethod
    def from_list(cls: type[SpanType], data: Sequence[SpanType | LoadFromType]) -> "SpanList[SpanType]":
        """Creates a list of span instances from a sequence of spans and/or primitive types.

        Args:
            data (list): List of span instances or objects of types they can be initialized from (default: dict).

        Returns:
            A list of span instances.
        """
        out = SpanList()
        for item in data:
            if isinstance(item, BaseSpan):
                out.append(item)
            elif isinstance(item, cls._load_from_type):
                out.append(cls.load(item))
            else:
                raise TypeError(
                    f"Invalid input type. Expected {str(cls._load_from_type)} or object inheriting from "
                    f"BaseSpan, got {type(item)}."
                )
        return out


@dataclass
class Span(BaseSpan[DictSpan]):
    """Class representing a span in a text.

    Attributes:
        start (int): The starting index of the span.
        end (int): The ending index of the span.
        label (str | int | float | None): The label of the span.
        text (str | None): The text of the span. Defaults to None.
    """

    start: int
    end: int
    label: str | int | float | None
    text: str | None = None


@dataclass
class EditSpan(BaseSpan[tuple[DictSpan, DictSpan]]):
    """Class representing a pair of spans connecting a text with an edit.
    Attributes:
        orig (Span): The span over the original text.
        edit (Span): The span over the edited text.
    """

    orig: Span
    edit: Span
    _load_from_type: type = tuple

    @classmethod
    def load(cls, data: tuple[DictSpan, DictSpan]) -> "EditSpan":
        orig = Span.load(data[0]) if isinstance(data[0], dict) else data[0]
        edit = Span.load(data[1]) if isinstance(data[1], dict) else data[1]
        return cls(orig, edit)


class SpanList(list[SpanType]):
    """Class for a list of `Span`, with custom visualization."""

    def __str__(self):
        return pformat(self, indent=4)


class ListOfListsOfSpans(list[SpanList[SpanType]]):
    """Class for a list of lists of `Span`, with custom visualization."""

    def __str__(self) -> str:
        return "\n".join(str(lst) for lst in self)
