from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

DictSpan = dict[str, str | int | float | None]
QESpanInput = Sequence["QESpan"] | Sequence[DictSpan]
QESpanWithEditInput = Sequence["QESpanWithEdit"] | Sequence[tuple[DictSpan, DictSpan]]

SpanType = TypeVar("SpanType", bound="AbstractSpan")
LoadFromType = TypeVar("LoadFromType", bound=DictSpan | tuple[DictSpan, DictSpan])


class AbstractSpan(Generic[LoadFromType]):
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
            k: v if not isinstance(v, AbstractSpan) else v.to_dict()
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
    def from_list(cls: type[SpanType], data: Sequence[SpanType | LoadFromType]) -> list[SpanType]:
        """Creates a list of span instances from a sequence of spans and/or primitive types.

        Args:
            data (list): List of span instances or objects of types they can be initialized from (default: dict).

        Returns:
            A list of span instances.
        """
        out = []
        for item in data:
            if isinstance(item, AbstractSpan):
                out.append(item)
            elif isinstance(item, cls._load_from_type):
                out.append(cls.load(item))
            else:
                raise TypeError(
                    f"Invalid input type. Expected {str(cls._load_from_type)} or object inheriting from "
                    f"AbstractSpan, got {type(item)}."
                )
        return out


@dataclass
class QESpan(AbstractSpan[DictSpan]):
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
class QESpanWithEdit(AbstractSpan[tuple[DictSpan, DictSpan]]):
    """Class representing a pair of spans connecting a text with an edit.
    Attributes:
        orig (QESpan): The span over the original text.
        edit (QESpan): The span over the edited text.
    """

    orig: QESpan
    edit: QESpan
    _load_from_type: type = tuple

    @classmethod
    def load(cls, data: tuple[DictSpan, DictSpan]) -> "QESpanWithEdit":
        orig = QESpan.load(data[0]) if isinstance(data[0], dict) else data[0]
        edit = QESpan.load(data[1]) if isinstance(data[1], dict) else data[1]
        return cls(orig, edit)
