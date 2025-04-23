from collections.abc import Sequence
from typing import Literal

from krippendorff.krippendorff import LevelOfMeasurement
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from wqe.data.base_dataset import BaseDataset
from wqe.data.labeled_entry import LabeledEntry
from wqe.utils.span import Span
from wqe.utils.token import LabelType
from wqe.utils.tokenizer import Tokenizer, get_tokenizer

CorrelationMethod = Literal["pearson", "spearman"]


class LabeledDataset(BaseDataset[LabeledEntry]):
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
                for idx in tqdm(range(len(tokens)), desc="Creating WQEDataset", total=len(tokens), unit="entries")
            ]
        )

    ### Utils ###

    def get_label_agreement(
        self,
        other: "LabeledDataset",
        level_of_measurement: LevelOfMeasurement | None = None,
        correlation_method: CorrelationMethod | None = None,
    ) -> float:
        """Compute the inter-annotator agreement for token labels between two datasets.

        The behavior of this function depends on the type of labels in the dataset:

        - If the labels are strings, it returns a list of lists, where element `[i][j]` is the agreement between
        annotator i and annotator j over the full dataset computed using [Krippendorff's alpha](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha).

        - If the labels are floats, it returns a list of lists, where element `[i][j]` is the correlation between
        annotator i and annotator j over the full dataset computed using [Pearson's](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) or
        [Spearman's](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) correlation.

        If the labels are integers, a `level_of_measurement` or a `correlation_method` must be specified to clarify whether
        to treat labels as categorical or numeric.

        Args:
            level_of_measurement (Literal['nominal', 'ordinal', 'interval', 'ratio']): The level of measurement for the
                labels when using Krippendorff's alpha. Can be "nominal", "ordinal", "interval", or "ratio", depending
                on the type of labels. Default: "nominal".
            correlation_method (CorrelationMethod): The correlation method to use when comparing numeric . Can be "pearson" or "spearman".
                Default: "spearman".

        Returns:
            The inter-annotator agreement between the two datasets.
        """
        if len(self.label_types) > 1:
            raise RuntimeError(
                f"Multiple label types found for dataset entries: {','.join(str(t) for t in self.label_types)}.\n"
                "A single label type should be present to compute inter-annotator agreement (for str or int discrete "
                "labels) or correlation (for numeric data). Transform the annotations using `data.relabel` to ensure "
                "a single type is present."
            )
        return 1.0  # TODO: Implement this function
