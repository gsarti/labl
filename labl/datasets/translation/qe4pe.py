from typing import Any, Literal, cast

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils.import_utils import is_datasets_available, is_pandas_available

from labl.data.edited_dataset import EditedDataset
from labl.utils.tokenizer import Tokenizer

Qe4peTask = Literal["pretask", "main", "posttask"]
Qe4peLanguage = Literal["ita", "nld"]
Qe4peDomain = Literal["biomedical", "social"]
Qe4peSpeedGroup = Literal["faster", "avg", "slower"]
Qe4peHighlightModality = Literal["no_highlight", "oracle", "supervised", "unsupervised"]

SPEED_MAP = {"faster": "t1", "avg": "t2", "slower": "t3"}


def load_qe4pe(
    configs: Qe4peTask | list[Qe4peTask] = "main",
    langs: Qe4peLanguage | list[Qe4peLanguage] = ["ita", "nld"],
    domains: Qe4peDomain | list[Qe4peDomain] | None = None,
    speed_groups: Qe4peSpeedGroup | list[Qe4peSpeedGroup] | None = None,
    highlight_modalities: Qe4peHighlightModality | list[Qe4peHighlightModality] | None = None,
    tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
    tokenizer_kwargs: dict[str, Any] = {},
    filter_issues: bool = True,
    with_gaps: bool = True,
    sub_label: str = "S",
    ins_label: str = "I",
    del_label: str = "D",
    gap_token: str = "▁",
) -> dict[str, dict[str, EditedDataset]]:
    """Load the QE4PE dataset by [Sarti et al. (2025)](https://arxiv.org/abs/2503.03044), containing multiple edits
        over a single set of machine-translated sentences in two languages (Italian and Dutch).

    Args:
        configs (Literal["pretask", "main", "posttask"] | list[Literal["pretask", "main", "posttask"]], *optional*):
            One or more task configurations to load. Defaults to "main".
            Available options: "pretask", "main", "posttask".
        langs (Literal["ita", "nld"] | list[Literal["ita", "nld"]], *optional*):
            One or more languages to load. Defaults to ["ita", "nld"].
            Available options: "ita", "nld".
        domains (Literal["biomedical", "social"] | list[Literal["biomedical", "social"]] | None, *optional*):
            One or more text categories to load. Defaults to ["biomedical", "social"].
            Available options: "biomedical", "social".
        speed_groups (Literal["faster", "avg", "slower"] | list[Literal["faster", "avg", "slower"]] | None, *optional*):
            One or more translator speed groups to load. Defaults to ["faster", "avg", "slower"].
            Available options: "faster", "avg", "slower".
        highlight_modalities (Literal["no_highlight", "oracle", "supervised", "unsupervised"] | list[Literal["no_highlight", "oracle", "supervised", "unsupervised"]] | None, *optional*):
            One or more highlight modalities to load. Defaults to all modalities.
            Available options: "no_highlight", "oracle", "supervised", "unsupervised".
        filter_issues (bool, *optional*):
            Whether to filter out issues from the dataset. Defaults to True.
        tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast, *optional*):
            The tokenizer to use for tokenization. If None, a default whitespace tokenizer will be used.
        tokenizer_kwargs (dict[str, Any], *optional*):
            Additional arguments for the tokenizer.
        with_gaps (bool, *optional*):
            Whether to include gaps in the tokenization. Defaults to True.
        sub_label (str, *optional*):
            The label for substitutions. Defaults to "S".
        ins_label (str, *optional*):
            The label for insertions. Defaults to "I".
        del_label (str, *optional*):
            The label for deletions. Defaults to "D".
        gap_token (str, *optional*):
            The token used for gaps. Defaults to "▁".

    Returns:
        A dictionary containing the loaded datasets for each task and language.
            The keys are the task configurations, and the values are dictionaries with language keys
            and `EditedDataset` objects as values. E.g. `load_qe4pe_dataset()["main"]["ita"]` returns the
            `EditedDataset` for the main task for Italian.
    """
    if not is_datasets_available() or not is_pandas_available():
        raise RuntimeError("The `datasets` library is not installed. Please install it to use this function.")
    import pandas as pd

    from datasets import DatasetDict, load_dataset

    if isinstance(configs, str):
        configs = [configs]
    if isinstance(langs, str):
        langs = [langs]
    if domains is None:
        domains = ["biomedical", "social"]
    if isinstance(domains, str):
        domains = [domains]
    if speed_groups is None:
        speed_groups = ["faster", "avg", "slower"]
    if isinstance(speed_groups, str):
        speed_groups = [speed_groups]
    if highlight_modalities is None:
        highlight_modalities = ["no_highlight", "oracle", "supervised", "unsupervised"]
    if isinstance(highlight_modalities, str):
        highlight_modalities = [highlight_modalities]
    out_dict = {}
    for config in configs:
        dataset = cast(DatasetDict, load_dataset("gsarti/qe4pe", config))
        df = cast(pd.DataFrame, dataset["train"].to_pandas())
        if filter_issues:
            df = df[(~df["has_issue"]) & (df["translator_main_id"] != "no_highlight_t4")]
        out_dict[config] = {}
        for lang in langs:
            print(f"Loading {config} task for eng->{lang}...")
            lang_df = df[(df["tgt_lang"] == lang) & df["wmt_category"].isin(domains)]
            lang_df = lang_df[
                lang_df["translator_main_id"].str.endswith(tuple(SPEED_MAP[g] for g in speed_groups))
                | lang_df["highlight_modality"].isin(highlight_modalities)
            ]
            labl_dataset = EditedDataset.from_edits_dataframe(
                lang_df,
                text_column="mt_text",
                edit_column="pe_text",
                entry_ids=["doc_id", "segment_in_doc_id"],
                infos_columns=[
                    "wmt_category",
                    "doc_id",
                    "segment_in_doc_id",
                    "translator_main_id",
                    "highlight_modality",
                ],
                tokenizer=tokenizer,
                tokenizer_kwargs=tokenizer_kwargs,
                with_gaps=with_gaps,
                sub_label=sub_label,
                ins_label=ins_label,
                del_label=del_label,
                gap_token=gap_token,
            )
            out_dict[config][lang] = labl_dataset
    return out_dict
