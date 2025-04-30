from collections.abc import Sequence

from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from labl.data.base_sequence import BaseLabeledDataset
from labl.data.labeled_entry import LabeledEntry
from labl.utils.span import Span
from labl.utils.tokenizer import Tokenizer, get_tokenizer
from labl.utils.typing import InfoDictType, LabelType, OffsetType, SpanType


class LabeledDataset(BaseLabeledDataset[LabeledEntry]):
    """Dataset class for handling collections of `LabeledEntry` objects.

    Attributes:
        data (list[LabeledEntry]): A list of LabeledEntry objects.
    """

    ### Constructors ###

    @classmethod
    def from_spans(
        cls,
        texts: list[str],
        spans: list[list[Span]] | list[list[SpanType]],
        infos: list[InfoDictType] | None = None,
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
        show_progress: bool = True,
    ) -> "LabeledDataset":
        """Create a `LabeledDataset` from a set of texts and one or more spans for each text.

        Args:
            texts (list[str]):
                The set of text.
            spans (list[list[Span]] | list[list[dict[str, str | int | float | None]]]):
                A list of spans for each text.
            infos (list[dict[str, str | int | float | bool]] | None):
                A list of dictionaries containing additional information for each entry.
                If None, no additional information is added. Defaults to None.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
            show_progress (bool): Whether to show a progress bar. Defaults to True.
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        if infos is None:
            infos = [{}] * len(texts)
        return cls(
            [
                LabeledEntry.from_spans(
                    text,
                    span,
                    tokenizer=tokenizer,
                    info=info,
                )
                for text, span, info in tqdm(
                    zip(texts, spans, infos, strict=True),
                    desc="Creating labeled dataset",
                    total=len(texts),
                    unit="entries",
                    disable=not show_progress,
                )
            ]
        )

    @classmethod
    def from_tagged(
        cls,
        tagged: list[str],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        keep_tags: list[str] = [],
        ignore_tags: list[str] = [],
        tokenizer_kwargs: dict = {},
        infos: list[InfoDictType] | None = None,
        show_progress: bool = True,
    ) -> "LabeledDataset":
        """Create a `LabeledDataset` from a set of tagged texts.

        Args:
            tagged (list[str]):
                The set of tagged text.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            infos (list[dict[str, str | int | float | bool]] | None):
                A list of dictionaries containing additional information for each entry.
                If None, no additional information is added. Defaults to None.
            keep_tags (list[str]): A list of tags to keep.
            ignore_tags (list[str]): A list of tags to ignore.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
            show_progress (bool): Whether to show a progress bar. Defaults to True.
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        if infos is None:
            infos = [{}] * len(tagged)
        return cls(
            [
                LabeledEntry.from_tagged(
                    text,
                    tokenizer=tokenizer,
                    keep_tags=keep_tags,
                    ignore_tags=ignore_tags,
                    info=info,
                )
                for text, info in tqdm(
                    zip(tagged, infos, strict=True),
                    desc="Creating labeled dataset",
                    total=len(tagged),
                    unit="entries",
                    disable=not show_progress,
                )
            ]
        )

    @classmethod
    def from_tokens(
        cls,
        tokens: list[list[str]],
        labels: Sequence[Sequence[LabelType]],
        texts: Sequence[str] | None = None,
        offsets: Sequence[list[OffsetType]] | None = None,
        infos: list[InfoDictType] | None = None,
        keep_labels: list[str] = [],
        ignore_labels: list[str] = [],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
        show_progress: bool = True,
    ) -> "LabeledDataset":
        """Create a `LabeledDataset` from a set of tokenized texts.

        Args:
            tokens (list[list[str]] | None):
                A list of lists of string tokens.
            labels (list[list[str | int | float | None]] | None):
                A list of lists of labels for the tokens.
            texts (list[str] | None):
                A list of texts corresponding to the tokens.
            offsets (list[list[OffsetType]] | None):
                A list of lists of offsets for the tokens.
            infos (list[dict[str, str | int | float | bool]] | None):
                A list of dictionaries containing additional information for each entry.
                If None, no additional information is added. Defaults to None.
            keep_labels (list[str]): A list of labels to keep.
            ignore_labels (list[str]): A list of labels to ignore.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
            show_progress (bool): Whether to show a progress bar. Defaults to True.
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        if infos is None:
            infos = [{}] * len(tokens)
        return cls(
            [
                LabeledEntry.from_tokens(
                    tokens=tokens[idx],
                    labels=labels[idx],
                    text=texts[idx] if texts is not None else None,
                    offsets=offsets[idx] if offsets is not None else None,
                    keep_labels=keep_labels,
                    ignore_labels=ignore_labels,
                    tokenizer=tokenizer,
                    info=infos[idx],
                )
                for idx in tqdm(
                    range(len(tokens)),
                    desc="Creating LabeledDataset",
                    total=len(tokens),
                    unit="entries",
                    disable=not show_progress,
                )
            ]
        )
