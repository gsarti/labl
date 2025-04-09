from typing import Any, cast

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.utils import is_pandas_available

from wqe.data.span import Span
from wqe.data.token import LabeledTokenInput
from wqe.data.tokenizer import Tokenizer, get_tokenizer
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
        with_gaps: bool = True,
        sub_label: str = "S",
        ins_label: str = "I",
        del_label: str = "D",
        gap_token: str = "▁",
    ) -> "WQEDataset":
        """Create a `WQEDataset` from a set of texts and one or more edits for each text.

        Args:
            texts (list[str]):
                The set of text.
            edits (list[str] | list[list[str]] | None):
                One or more edited version for each text.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
            with_gaps (bool): Whether to add gaps to the tokens and offsets. Gaps are used to mark the positions of
                insertions and deletions in the original/edited texts, respectively. If false, those are merged to the
                next token to the right. Default: True.
            sub_label (str): The label for substitutions. Default: "S".
            ins_label (str): The label for insertions. Default: "I".
            del_label (str): The label for deletions. Default: "D".
            gap_token (str): The token to use for gaps. Default: "▁".
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        return cls(
            [
                WQEEntry.from_edits(
                    text,
                    edit,
                    tokenizer=tokenizer,
                    with_gaps=with_gaps,
                    sub_label=sub_label,
                    ins_label=ins_label,
                    del_label=del_label,
                    gap_token=gap_token,
                )
                for text, edit in tqdm(
                    zip(texts, edits, strict=True), desc="Creating WQEDataset", total=len(texts), unit="entries"
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
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        return cls(
            [
                WQEEntry.from_spans(
                    text,
                    span,
                    tokenizer=tokenizer,
                )
                for text, span in tqdm(
                    zip(texts, spans, strict=True), desc="Creating WQEDataset", total=len(texts), unit="entries"
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
                WQEEntry.from_tagged(
                    text,
                    tokenizer=tokenizer,
                    keep_tags=keep_tags,
                    ignore_tags=ignore_tags,
                )
                for text in tqdm(tagged, desc="Creating WQEDataset", total=len(tagged), unit="entries")
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
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
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
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        return cls(
            [
                WQEEntry.from_tokens(
                    labeled_tokens=labeled_tokens[idx] if labeled_tokens is not None else None,  # type: ignore
                    keep_labels=keep_labels,
                    ignore_labels=ignore_labels,
                    tokenizer=tokenizer,
                    tokens=tokens[idx] if tokens is not None else None,
                    labels=labels[idx] if labels is not None else None,
                )
                for idx in tqdm(range(num_sequences), desc="Creating WQEDataset", total=num_sequences, unit="entries")
            ]
        )

        ### Loaders ###

    @classmethod
    def from_edits_dataframe(
        cls,
        df,
        text_column: str,
        edit_column: str,
        entry_ids: str | list[str],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict[str, Any] = {},
        with_gaps: bool = True,
        sub_label: str = "S",
        ins_label: str = "I",
        del_label: str = "D",
        gap_token: str = "▁",
    ) -> "WQEDataset":
        """Create a `WQEDataset` from a `pandas.DataFrame` with edits.

        Every row in the DataFrame is an entry identified univocally by `entry_ids`. The `text_column` contains the
        original text, and the `edit_column` contains the edits. If multiple columns with the same `entry_ids` are
        present, they are all treated as edits of the same text.

        Args:
            df (pandas.DataFrame): The DataFrame containing the text and edits.
            text_column (str): The name of the column in the dataframe containing the original text.
            edit_column (str): The name of the column in the dataframe containing the edited text.
            entry_ids (str | list[str]): One or more column names acting as unique identifiers for each entry. If
                multiple entries are found with the same `entry_ids`, they are all treated as edits of the same text.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None, optional): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict[str, Any], optional): _description_. Defaults to {}.
            with_gaps (bool): Whether to add gaps to the tokens and offsets. Gaps are used to mark the positions of
                insertions and deletions in the original/edited texts, respectively. If false, those are merged to the
                next token to the right. Default: True.
            sub_label (str): The label for substitutions. Default: "S".
            ins_label (str): The label for insertions. Default: "I".
            del_label (str): The label for deletions. Default: "D".
            gap_token (str): The token to use for gaps. Default: "▁".

        Returns:
            A `WQEDataset` initialized from the set of texts and edits.
        """
        if not is_pandas_available():
            raise ImportError("Pandas is not installed. Please install pandas to use this function.")
        import pandas as pd

        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        df = cast(pd.DataFrame, df)
        grouped_dfs = df.groupby(entry_ids).size().reset_index()
        all_texts = []
        all_edits = []
        for _, entry_row in tqdm(
            grouped_dfs.iterrows(), desc="Extracting texts and edits", total=len(grouped_dfs), unit="entries"
        ):
            curr_vals = [entry_row[col] for col in entry_ids]
            selected_rows = df[(df[entry_ids] == curr_vals).all(axis=1)]
            text = selected_rows[text_column].tolist()[0]
            edits = selected_rows[edit_column].tolist()
            all_texts.append(text)
            all_edits.append(edits)
        return WQEDataset.from_edits(
            all_texts,
            all_edits,
            tokenizer=tokenizer,
            with_gaps=with_gaps,
            sub_label=sub_label,
            ins_label=ins_label,
            del_label=del_label,
            gap_token=gap_token,
        )
