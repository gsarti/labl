from collections.abc import Callable, Sequence
from typing import Any, cast

from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils.import_utils import is_pandas_available

from labl.data.base_sequence import BaseLabeledDataset
from labl.data.edited_entry import EditedEntry, MultiEditEntry
from labl.utils.tokenizer import Tokenizer, get_tokenizer
from labl.utils.typing import InfoDictType, LabelType


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
        infos: list[InfoDictType] | list[list[InfoDictType]] | None = None,
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
        with_gaps: bool = True,
        keep_final_gap: bool = True,
        sub_label: str = "S",
        ins_label: str = "I",
        del_label: str = "D",
        gap_token: str = "▁",
        show_progress: bool = True,
    ) -> "EditedDataset":
        """Create an `EditedDataset` from a set of texts and one or more edits for each text.

        Args:
            texts (list[str]):
                The set of text.
            edits (list[str] | list[list[str]] | None):
                One or more edited version for each text.
            infos (list[dict[str, str | int | float | bool]] | list[list[dict[str, str | int | float | bool]]] | None):
                A list of dictionaries containing additional information for each entry.
                If multiple edits are provided for each text, `infos` can be a list of lists of dictionaries (one per
                edit per entry). If None, no additional information is added. Defaults to None.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
            with_gaps (bool): Whether to add gaps to the tokens and offsets. Gaps are used to mark the positions of
                insertions and deletions in the original/edited texts, respectively. If false, those are merged to the
                next token to the right. Default: True.
            keep_final_gap (bool): Whether to keep the final gap when merging gaps to account for end insertions.
                If false, information about end insertion is lost. Default: True.
            sub_label (str): The label for substitutions. Default: "S".
            ins_label (str): The label for insertions. Default: "I".
            del_label (str): The label for deletions. Default: "D".
            gap_token (str): The token to use for gaps. Default: "▁".
            show_progress (bool): Whether to show a progress bar. Default: True.
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        if infos is None:
            infos = [{}] * len(texts)
        return cls(
            [
                EditedEntry.from_edits(
                    text=text,
                    edits=edit,
                    tokenizer=tokenizer,
                    with_gaps=with_gaps,
                    keep_final_gap=keep_final_gap,
                    sub_label=sub_label,
                    ins_label=ins_label,
                    del_label=del_label,
                    gap_token=gap_token,
                    info=info,
                )
                for text, edit, info in tqdm(
                    zip(texts, edits, infos, strict=True),
                    desc="Creating EditedDataset",
                    total=len(texts),
                    unit="entries",
                    disable=not show_progress,
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
        infos_columns: list[str] = [],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict[str, Any] = {},
        with_gaps: bool = True,
        keep_final_gap: bool = True,
        sub_label: str = "S",
        ins_label: str = "I",
        del_label: str = "D",
        gap_token: str = "▁",
        show_progress: bool = True,
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
            infos_columns (list[str]): A list of columns containing additional information for each entry.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None, optional): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict[str, Any], optional): _description_. Defaults to {}.
            with_gaps (bool): Whether to add gaps to the tokens and offsets. Gaps are used to mark the positions of
                insertions and deletions in the original/edited texts, respectively. If false, those are merged to the
                next token to the right. Default: True.
            keep_final_gap (bool): Whether to keep the final gap when merging gaps to account for end insertions.
                If false, information about end insertion is lost. Default: True.
            sub_label (str): The label for substitutions. Default: "S".
            ins_label (str): The label for insertions. Default: "I".
            del_label (str): The label for deletions. Default: "D".
            gap_token (str): The token to use for gaps. Default: "▁".
            show_progress (bool): Whether to show a progress bar. Default: True.

        Returns:
            An `EditedDataset` initialized from the set of texts and edits.
        """
        if not is_pandas_available():
            raise ImportError("Pandas is not installed. Please install pandas to use this function.")
        import pandas as pd

        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        df = cast(pd.DataFrame, df)
        if isinstance(entry_ids, str):
            entry_ids = [entry_ids]
        grouped_dfs = df.groupby(entry_ids).size().reset_index()
        all_texts = []
        all_edits = []
        all_infos = []
        for _, entry_row in grouped_dfs.iterrows():
            curr_vals = [entry_row[col] for col in entry_ids]
            edit_rows = df[(df[entry_ids] == curr_vals).all(axis=1)]
            text = edit_rows[text_column].tolist()[0]
            edits = edit_rows[edit_column].tolist()
            all_texts.append(text)
            all_edits.append(edits)
            infos = []
            for _, edit_row in edit_rows.iterrows():
                infos.append({col: edit_row[col] for col in infos_columns})
            all_infos.append(infos)
        return EditedDataset.from_edits(
            all_texts,
            all_edits,
            all_infos,
            tokenizer=tokenizer,
            with_gaps=with_gaps,
            keep_final_gap=keep_final_gap,
            sub_label=sub_label,
            ins_label=ins_label,
            del_label=del_label,
            gap_token=gap_token,
            show_progress=show_progress,
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
