from typing import Any, Literal, cast

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils.import_utils import is_datasets_available, is_pandas_available

from labl.data.edited_dataset import EditedDataset
from labl.utils.tokenizer import Tokenizer
from labl.utils.typing import to_list

Qe4peTask = Literal["oracle_pe", "pretask", "main", "posttask"]
Qe4peLanguage = Literal["ita", "nld"]
Qe4peDomain = Literal["biomedical", "social"]
Qe4peSpeedGroup = Literal["faster", "avg", "slower"]
Qe4peHighlightModality = Literal["no_highlight", "oracle", "supervised", "unsupervised"]

SPEED_MAP = {"faster": "t1", "avg": "t2", "slower": "t3"}


def load_qe4pe(
    configs: Qe4peTask | list[Qe4peTask] | None = None,
    langs: Qe4peLanguage | list[Qe4peLanguage] | None = None,
    domains: Qe4peDomain | list[Qe4peDomain] | None = None,
    speed_groups: Qe4peSpeedGroup | list[Qe4peSpeedGroup] | None = None,
    highlight_modalities: Qe4peHighlightModality | list[Qe4peHighlightModality] | None = None,
    tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
    tokenizer_kwargs: dict[str, Any] = {},
    with_gaps: bool = True,
    keep_final_gap: bool = True,
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
        tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast, *optional*):
            The tokenizer to use for tokenization. If None, a default whitespace tokenizer will be used.
        tokenizer_kwargs (dict[str, Any], *optional*):
            Additional arguments for the tokenizer.
        with_gaps (bool, *optional*):
            Whether to include gaps in the tokenization. Defaults to True.
        keep_final_gap (bool): Whether to keep the final gap when merging gaps to account for end insertions.
                If false, information about end insertion is lost. Default: True.
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

    configs = to_list(configs, ["main"])
    langs = to_list(langs, ["ita", "nld"])
    domains = to_list(domains, ["biomedical", "social"])
    speed_groups = to_list(speed_groups, ["faster", "avg", "slower"])
    highlight_modalities = to_list(highlight_modalities, ["no_highlight", "oracle", "supervised", "unsupervised"])
    out_dict = {}
    for config in configs:
        dataset = cast(DatasetDict, load_dataset("gsarti/qe4pe", config))
        df = cast(pd.DataFrame, dataset["train"].to_pandas())
        out_dict[config] = {}
        for lang in langs:
            print(f"Loading {config} task for eng->{lang}...")
            lang_df = df[(df["tgt_lang"] == lang) & df["wmt_category"].isin(domains)]
            lang_df = lang_df[
                lang_df["translator_main_id"].str.endswith(tuple(SPEED_MAP[g] for g in speed_groups))
                & lang_df["highlight_modality"].isin(highlight_modalities)
            ]
            infos_columns = [
                "src_text",
                "has_issue",
                "wmt_category",
                "doc_id",
                "segment_in_doc_id",
                "translator_main_id",
                "highlight_modality",
            ]
            if config == "main":
                infos_columns += ["qa_mt_annotator_id", "qa_mt_esa_rating", "qa_pe_annotator_id", "qa_pe_esa_rating"]
            labl_dataset = EditedDataset.from_edits_dataframe(
                lang_df,
                text_column="mt_text",
                edit_column="pe_text",
                entry_ids=["doc_id", "segment_in_doc_id"],
                infos_columns=infos_columns,
                tokenizer=tokenizer,
                tokenizer_kwargs=tokenizer_kwargs,
                with_gaps=with_gaps,
                keep_final_gap=keep_final_gap,
                sub_label=sub_label,
                ins_label=ins_label,
                del_label=del_label,
                gap_token=gap_token,
            )
            out_dict[config][lang] = labl_dataset
    return out_dict
