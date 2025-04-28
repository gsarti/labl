from collections.abc import Sequence
from dataclasses import asdict, dataclass

from labl.utils.typing import LabelType


@dataclass
class Span:
    """Class representing a span in a text.

    Attributes:
        start (int): The starting index of the span.
        end (int): The ending index of the span.
        label (str | int | float | None): The label of the span.
        text (str | None): The text of the span. Defaults to None.
    """

    start: int
    end: int
    label: LabelType
    text: str | None = None

    def __str__(self) -> str:
        """Returns a string representation of the span."""
        return f"{self.start}:{self.end} ({self.text}) => {self.label}"

    def to_dict(self) -> dict[str, LabelType]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Span":
        return cls(**data)

    @classmethod
    def from_list(cls, data: "Sequence[Span | dict]") -> "SpanList":
        """Creates a list of span instances from a sequence of spans and/or primitive types.

        Args:
            data (list): List of span instances or dictionaries from which they can be initialized.

        Returns:
            A list of span instances.
        """
        out = SpanList()
        for item in data:
            if isinstance(item, cls):
                out.append(item)
            elif isinstance(item, dict):
                out.append(cls.from_dict(item))
            else:
                raise TypeError("Invalid input type")
        return out


class SpanList(list[Span]):
    """Class for a list of `Span`, with custom visualization."""

    def __str__(self):
        return "\n".join(f"{idx}: {span}" for idx, span in enumerate(self))
