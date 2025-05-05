from typing import Any, Literal, cast

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils.import_utils import is_datasets_available, is_pandas_available

from labl.data.edited_dataset import EditedDataset
from labl.utils.tokenizer import Tokenizer
from labl.utils.typing import to_list

DivemtTask = Literal["warmup", "main"]
DivemtLanguage = Literal["ara", "nld", "ita", "tur", "ukr", "vie"]
DivemtMTModel = Literal["gtrans", "mbart50"]

MT_MODEL_MAP = {"gtrans": "pe1", "mbart50": "pe2"}


def load_divemt(
    configs: DivemtTask | list[DivemtTask] | None = None,
    langs: DivemtLanguage | list[DivemtLanguage] | None = None,
    mt_models: DivemtMTModel | list[DivemtMTModel] | None = None,
    tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
    tokenizer_kwargs: dict[str, Any] = {},
    with_gaps: bool = True,
    keep_final_gap: bool = True,
    sub_label: str = "S",
    ins_label: str = "I",
    del_label: str = "D",
    gap_token: str = "▁",
) -> dict[str, dict[str, dict[str, EditedDataset]]]:
    """Load the DivEMT dataset by [Sarti et al. (2022)](https://aclanthology.org/2022.emnlp-main.532/), containing edits
        over two sets of machine-translated sentences across six typologically diverse languages.

    Args:
        configs (Literal["warmup", "main"] | list[Literal["warmup", "main"]], *optional*):
            One or more task configurations to load. Defaults to "main".
            Available options: "warmup", "main".
        langs (Literal["ara", "nld", "ita", "tur", "ukr", "vie"] | list[Literal["ara", "nld", "ita", "tur", "ukr", "vie"]], *optional*):
            One or more languages to load. Defaults to None (all languages).
            Available options: "ara", "nld", "ita", "tur", "ukr", "vie".
        mt_models (Literal["gtrans", "mbart50"] | list[Literal["gtrans", "mbart50"]], *optional*):
            One or more models for which post-edits need to be loaded. Defaults to None (all models).
            Available options: "gtrans", "mbart50".
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
        A dictionary containing the loaded datasets for each task, language, and MT model.
            The keys are the task configurations, and the values are dictionaries with language keys
            and `EditedDataset` objects as values. E.g. `load_divemt_dataset()["main"]["ita"]["mbart50"]` returns the
            `EditedDataset` for the main task for Italian.
    """
    if not is_datasets_available() or not is_pandas_available():
        raise RuntimeError("The `datasets` library is not installed. Please install it to use this function.")
    import pandas as pd

    from datasets import DatasetDict, load_dataset

    configs = to_list(configs, ["main"])
    langs = to_list(langs, ["ara", "nld", "ita", "tur", "ukr", "vie"])
    mt_models = to_list(mt_models, ["gtrans", "mbart50"])
    out_dict = {}
    for config in configs:
        dataset = cast(DatasetDict, load_dataset("GroNLP/divemt", config))
        df = cast(pd.DataFrame, dataset["train"].to_pandas())
        out_dict[config] = {}
        for lang in langs:
            out_dict[config][lang] = {}
            for model in mt_models:
                print(f"Loading {config} task for eng->{lang} {model} edits...")
                filter_df = df[(df["lang_id"] == lang) & (df["task_type"] == MT_MODEL_MAP[model])]
                labl_dataset = EditedDataset.from_edits_dataframe(
                    filter_df,
                    text_column="mt_text",
                    edit_column="tgt_text",
                    entry_ids="item_id",
                    infos_columns=["doc_id", "subject_id", "item_id", "src_text"],
                    tokenizer=tokenizer,
                    tokenizer_kwargs=tokenizer_kwargs,
                    with_gaps=with_gaps,
                    keep_final_gap=keep_final_gap,
                    sub_label=sub_label,
                    ins_label=ins_label,
                    del_label=del_label,
                    gap_token=gap_token,
                )
                out_dict[config][lang][model] = labl_dataset
    return out_dict
