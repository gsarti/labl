from collections.abc import Sequence
from typing import TypedDict, TypeVar

LabelType = str | int | float | None
OffsetType = tuple[int, int] | None
SpanType = dict[str, LabelType]
InfoDictType = dict[str, str | int | float | bool]

T = TypeVar("T")


class SerializableDictType(TypedDict):
    _class: str
    info: InfoDictType


class EntrySequenceDictType(SerializableDictType):
    entries: Sequence[SerializableDictType]


class LabeledEntryDictType(SerializableDictType):
    text: str
    tagged: str
    tokens: list[str]
    tokens_labels: Sequence[LabelType]
    tokens_offsets: list[OffsetType]
    spans: list[SpanType]


class EditedEntryDictType(SerializableDictType):
    orig: LabeledEntryDictType
    edit: LabeledEntryDictType
    has_bos_token: bool
    has_eos_token: bool
    has_gaps: bool


def to_list(val: T | list[T] | None, default: list[T] = []) -> list[T]:
    if val is None:
        return default
    if isinstance(val, list):
        return list(val)
    return [val]
