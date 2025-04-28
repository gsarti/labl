from collections.abc import Sequence

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from labl.data.base_sequence import BaseLabeledDataset
from labl.data.labeled_entry import LabeledEntry
from labl.utils.span import Span
from labl.utils.token import LabelType
from labl.utils.tokenizer import Tokenizer, get_tokenizer


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
        spans: list[list[Span]] | list[list[dict[str, LabelType]]],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
    ) -> "LabeledDataset":
        """Create a `LabeledDataset` from a set of texts and one or more spans for each text.

        Args:
            texts (list[str]):
                The set of text.
            spans (list[list[Span]] | list[list[dict[str, str | int | float | None]]]):
                A list of spans for each text.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        return cls(
            [
                LabeledEntry.from_spans(
                    text,
                    span,
                    tokenizer=tokenizer,
                )
                for text, span in tqdm(
                    zip(texts, spans, strict=True), desc="Creating labeled dataset", total=len(texts), unit="entries"
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
    ) -> "LabeledDataset":
        """Create a `LabeledDataset` from a set of tagged texts.

        Args:
            tagged (list[str]):
                The set of tagged text.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            keep_tags (list[str]): A list of tags to keep.
            ignore_tags (list[str]): A list of tags to ignore.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        return cls(
            [
                LabeledEntry.from_tagged(
                    text,
                    tokenizer=tokenizer,
                    keep_tags=keep_tags,
                    ignore_tags=ignore_tags,
                )
                for text in tqdm(tagged, desc="Creating labeled dataset", total=len(tagged), unit="entries")
            ]
        )

    @classmethod
    def from_tokens(
        cls,
        tokens: list[list[str]],
        labels: Sequence[Sequence[LabelType]],
        keep_labels: list[str] = [],
        ignore_labels: list[str] = [],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
    ) -> "LabeledDataset":
        """Create a `LabeledDataset` from a set of tokenized texts.

        Args:
            tokens (list[list[str]] | None):
                A list of lists of string tokens.
            labels (list[list[str | int | float | None]] | None):
                A list of lists of labels for the tokens.
            keep_labels (list[str]): A list of labels to keep.
            ignore_labels (list[str]): A list of labels to ignore.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        return cls(
            [
                LabeledEntry.from_tokens(
                    tokens=tokens[idx],
                    labels=labels[idx],
                    keep_labels=keep_labels,
                    ignore_labels=ignore_labels,
                    tokenizer=tokenizer,
                )
                for idx in tqdm(range(len(tokens)), desc="Creating LabeledDataset", total=len(tokens), unit="entries")
            ]
        )
