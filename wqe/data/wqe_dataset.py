from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from wqe.data.span import Span
from wqe.data.token import LabeledTokenInput
from wqe.data.tokenizer import Tokenizer
from wqe.data.wqe_entry import WQEEntry


class WQEDataset:
    """Dataset class for word-level QE entries.

    Attributes:
        data (list[WQEEntry]): A list of WQEEntry objects.
    """

    def __init__(self, data: list[WQEEntry]):
        """Initialize the WQEDataset with a list of WQEEntry objects.

        Args:
            data (list[WQEEntry]): A list of WQEEntry objects.
        """
        self.data: list[WQEEntry] = data

    def __getitem__(self, idx) -> WQEEntry:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    def __add__(self, other: "WQEDataset") -> "WQEDataset":
        return WQEDataset(self.data + other.data)

    ### Constructors ###

    @classmethod
    def from_edits(
        cls,
        texts: list[str],
        edits: list[str] | list[list[str]],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
    ) -> "WQEDataset":
        """Create a `WQEDataset` from a set of texts and one or more edits for each text.

        Args:
            texts (list[str]):
                The set of text.
            edits (list[str] | list[list[str]] | None):
                One or more edited version for each text.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A
                [jiwer transform](https://jitsi.github.io/jiwer/reference/transforms/#transforms)
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
        """
        return cls(
            [
                WQEEntry.from_edits(
                    text,
                    edit,
                    tokenizer=tokenizer,
                    tokenizer_kwargs=tokenizer_kwargs,
                )
                for text, edit in tqdm(
                    zip(texts, edits, strict=True), desc="Creating WQEDataset", total=len(texts), unit="#"
                )
            ]
        )

    @classmethod
    def from_spans(
        cls,
        texts: list[str],
        spans: list[list[Span]] | list[list[dict[str, str | int | float | None]]],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
    ) -> "WQEDataset":
        """Create a `WQEDataset` from a set of texts and one or more spans for each text.

        Args:
            texts (list[str]):
                The set of text.
            spans (list[list[Span]] | list[list[dict[str, str | int | float | None]]]):
                A list of spans for each text.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A
                [jiwer transform](https://jitsi.github.io/jiwer/reference/transforms/#transforms)
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
        """
        return cls(
            [
                WQEEntry.from_spans(
                    text,
                    span,
                    tokenizer=tokenizer,
                    tokenizer_kwargs=tokenizer_kwargs,
                )
                for text, span in tqdm(
                    zip(texts, spans, strict=True), desc="Creating WQEDataset", total=len(texts), unit="#"
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
    ) -> "WQEDataset":
        """Create a `WQEDataset` from a set of tagged texts.

        Args:
            tagged (list[str]):
                The set of tagged text.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A
                [jiwer transform](https://jitsi.github.io/jiwer/reference/transforms/#transforms)
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            keep_tags (list[str]): A list of tags to keep.
            ignore_tags (list[str]): A list of tags to ignore.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
        """
        return cls(
            [
                WQEEntry.from_tagged(
                    text,
                    tokenizer=tokenizer,
                    keep_tags=keep_tags,
                    ignore_tags=ignore_tags,
                    tokenizer_kwargs=tokenizer_kwargs,
                )
                for text in tqdm(tagged, desc="Creating WQEDataset", total=len(tagged), unit="#")
            ]
        )

    @classmethod
    def from_tokens(
        cls,
        labeled_tokens: list[LabeledTokenInput] | None = None,
        keep_labels: list[str] = [],
        ignore_labels: list[str] = [],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
        tokens: list[list[str]] | None = None,
        labels: list[list[str | int | float | None]] | None = None,
    ) -> "WQEDataset":
        """Create a `WQEDataset` from a set of tokenized texts.

        Args:
            tokens (list[LabeledTokenInput]):
                A list of lists containing labeled tokens in the form of tuples (token, label) or `LabeledToken` objects.
            keep_labels (list[str]): A list of labels to keep.
            ignore_labels (list[str]): A list of labels to ignore.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A
                [jiwer transform](https://jitsi.github.io/jiwer/reference/transforms/#transforms)
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
            tokens (list[list[str]] | None):
                A list of lists of tokens. Can be provided together with `labels` as an alternative to `labeled_tokens`.
            labels (list[list[str | int | float | None]] | None):
                A list of lists of labels for the tokens. Can be provided together with `tokens` as an alternative to
                `labeled_tokens`.
        """
        if labeled_tokens is not None:
            num_sequences = len(labeled_tokens)
        elif tokens is not None and labels is not None:
            num_sequences = len(tokens)
        else:
            raise ValueError("Either `labeled_tokens` or both `tokens` and `labels` must be provided.")
        return cls(
            [
                WQEEntry.from_tokens(
                    labeled_tokens=labeled_tokens[idx] if labeled_tokens is not None else None,  # type: ignore
                    keep_labels=keep_labels,
                    ignore_labels=ignore_labels,
                    tokenizer=tokenizer,
                    tokenizer_kwargs=tokenizer_kwargs,
                    tokens=tokens[idx] if tokens is not None else None,
                    labels=labels[idx] if labels is not None else None,
                )
                for idx in tqdm(range(num_sequences), desc="Creating WQEDataset", total=num_sequences, unit="#")
            ]
        )
