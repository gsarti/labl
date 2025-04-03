from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from wqe.data.qe_span import QESpan
from wqe.data.tokenizer import LabeledTokenInput, Tokenizer
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
        spans: list[list[QESpan]] | list[list[dict[str, str | int | float | None]]],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
    ) -> "WQEDataset":
        """Create a `WQEDataset` from a set of texts and one or more spans for each text.

        Args:
            texts (list[str]):
                The set of text.
            spans (list[list[QESpan]] | list[list[dict[str, str | int | float | None]]]):
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
        tokens: list[LabeledTokenInput],
        keep_labels: list[str] = [],
        ignore_labels: list[str] = [],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
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
        """
        return cls(
            [
                WQEEntry.from_tokens(
                    text,
                    keep_labels=keep_labels,
                    ignore_labels=ignore_labels,
                    tokenizer=tokenizer,
                    tokenizer_kwargs=tokenizer_kwargs,
                )
                for text in tqdm(tokens, desc="Creating WQEDataset", total=len(tokens), unit="#")
            ]
        )
