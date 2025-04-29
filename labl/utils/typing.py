from collections.abc import Sequence
from typing import TypedDict

LabelType = str | int | float | None
OffsetType = tuple[int, int] | None
SpanType = dict[str, LabelType]


class SerializableDictType(TypedDict):
    _class: str


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
