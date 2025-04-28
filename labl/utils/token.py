from collections.abc import Sequence
from dataclasses import dataclass

from labl.utils.typing import LabelType

LabeledTokenInput = (
    list[tuple[str, str | None]]
    | list[tuple[str, int | None]]
    | list[tuple[str, float | None]]
    | list["LabeledToken"]
    | list[tuple[str, LabelType]]
)


@dataclass
class LabeledToken:
    """Class for a token with an associated label.

    Attributes:
        token (str): The token. Can be accessed using `.t`.
        label (str | int | float | None): The label associated with the token. Can be accessed using `.l`.
    """

    token: str
    label: LabelType

    def __str__(self):
        return f"({self.token}, {self.label})"

    @property
    def t(self):
        return self.token

    @property
    def l(self):  # noqa: E743
        return self.label

    def to_tuple(self):
        return (self.token, self.label)

    @classmethod
    def from_tuple(cls, tup: tuple[str, LabelType]) -> "LabeledToken":
        return cls(*tup)

    @classmethod
    def from_list(
        cls,
        lst: LabeledTokenInput,
        keep_labels: Sequence[LabelType] = [],
        ignore_labels: Sequence[LabelType] = [],
    ) -> "LabeledTokenList":
        out: LabeledTokenList = LabeledTokenList()
        for item in lst:
            if isinstance(item, tuple | list) and len(item) == 2:
                lab_token = cls.from_tuple(item)
            elif isinstance(item, LabeledToken):
                lab_token = item
            else:
                raise TypeError(f"Invalid type for LabeledToken input. Expected tuple, got {type(item)}.")
            is_ignored = lab_token.l in ignore_labels
            is_kept = not keep_labels or lab_token.l in keep_labels
            has_valid_label = is_kept and not is_ignored
            if not has_valid_label:
                lab_token = LabeledToken(lab_token.t, None)
            out.append(lab_token)
        return out


class LabeledTokenList(list[LabeledToken]):
    """Class for a list of `LabeledToken`, with custom visualization."""

    def __str__(self) -> str:
        lengths = [max(len(str(t.t)), len(str(t.l))) if t.l is not None else len(str(t.t)) for t in self]
        txt_toks = " ".join(f"{tok.t:>{tok_len}}" for tok, tok_len in zip(self, lengths, strict=True)) + "\n"
        txt_labels = (
            " ".join(
                f"{tok.l if tok.l is not None else '':>{tok_len}}" for tok, tok_len in zip(self, lengths, strict=True)
            )
            + "\n"
        )
        return txt_toks + txt_labels
