from collections.abc import Callable, Sequence
from typing import Any, cast

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.utils import is_pandas_available

from labl.data.base_sequence import BaseLabeledDataset
from labl.data.edited_entry import EditedEntry, MultiEditEntry
from labl.utils.tokenizer import Tokenizer, get_tokenizer
from labl.utils.typing import LabelType


class EditedDataset(BaseLabeledDataset[EditedEntry]):
    """Dataset class for handling collections of `EditedEntry` and `MultiEditEntry` objects.

    Attributes:
        data (list[EditedEntry] | list[MultiEditEntry]): A list of `EditedEntry` or `MultiEditEntry` objects.
    """

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
    ) -> "EditedDataset":
        """Create an `EditedDataset` from a set of texts and one or more edits for each text.

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
                EditedEntry.from_edits(
                    text=text,
                    edits=edit,
                    tokenizer=tokenizer,
                    with_gaps=with_gaps,
                    sub_label=sub_label,
                    ins_label=ins_label,
                    del_label=del_label,
                    gap_token=gap_token,
                )
                for text, edit in tqdm(
                    zip(texts, edits, strict=True), desc="Creating EditedDataset", total=len(texts), unit="entries"
                )
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
    ) -> "EditedDataset":
        """Create an `EditedDataset` from a `pandas.DataFrame` with edits.

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
            An `EditedDataset` initialized from the set of texts and edits.
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
        return EditedDataset.from_edits(
            all_texts,
            all_edits,
            tokenizer=tokenizer,
            with_gaps=with_gaps,
            sub_label=sub_label,
            ins_label=ins_label,
            del_label=del_label,
            gap_token=gap_token,
        )

    ### Utility functions ###

    def merge_gap_annotations(
        self,
        merge_fn: Callable[[Sequence[LabelType]], LabelType] | None = None,
        keep_final_gap: bool = True,
    ) -> None:
        """Merge gap annotations in the tokens of `orig` and `edit`.

        This method is equivalent to calling `EditedEntry.from_edits` with `with_gaps=False`. Gap annotations are merged
        to the next non-gap token to the right, and the gap label is added to the label of the non-gap token. The last
        gap is kept to account for insertions at the end of the text.

        E.g. `GAP Hello GAP World GAP ! GAP` becomes `Hello World ! GAP`.
             `  I     S   I               I`         `   IS     I     I`
        """
        for entry in self:
            cast(EditedEntry | MultiEditEntry, entry).merge_gap_annotations(
                merge_fn=merge_fn, keep_final_gap=keep_final_gap
            )
