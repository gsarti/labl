import json
from typing import Any, Literal

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils.import_utils import is_pandas_available

from labl.data.labeled_dataset import LabeledDataset
from labl.utils.cache import load_cached_or_download
from labl.utils.span import Span
from labl.utils.tokenizer import Tokenizer
from labl.utils.typing import to_list

D2TDataset = Literal["d2t-openweather", "d2t-football", "d2t-gsmarena",]
D2TNLGModel = Literal["gemma2", "gpt4o", "llama3-3", "phi3-5"]
D2TSplit = Literal["iaa", "test"]

def load_data2text(
    datasets: D2TDataset | list[D2TDataset] | None = None,
    splits: D2TSplit | list[D2TSplit] | None = None,
    nlg_models: D2TNLGModel | list[D2TNLGModel] | None = None,
    tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
    tokenizer_kwargs: dict[str, Any] = {},
) -> dict[str, dict[str, LabeledDataset]]:
    """Load the D2T span annotations from [Kasner et al. (2025)](https://arxiv.org/abs/2504.08697) over multiple NLG system outputs.

    Args:
        datasets (D2TDataset | list[D2TDataset] | None):
            One or more span-annotated datasets to load. Defaults to `["d2t-openweather", "d2t-football", "d2t-gsmarena"]`.
        datasets (D2TDataset | list[D2TDataset] | None):
            Which annotated split to load (`test` or `iaa`), with `test` being the main and default.
        nlg_models (NLGD2TModel | list[NLGD2TModel] | None):
            One or more models for which annotations need to be loaded. Defaults to `["gemma2", "gpt4o", "llama3-3", "phi3-5"]`.
        tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast, *optional*):
            The tokenizer to use for tokenization. If None, a default whitespace tokenizer will be used.
        tokenizer_kwargs (dict[str, Any], *optional*):
            Additional arguments for the tokenizer.

    Returns:
        A dictionary containing the loaded datasets for each MT model and language.
            The keys are the task configurations, and the values are dictionaries with language keys
            and `EditedDataset` objects as values. E.g. `load_wmt24esa()["Aya23"]["en-cs"]` returns
            the `LabeledDataset` for the Aya23 model for English-Czech.
    """

    if not is_pandas_available():
        raise RuntimeError("The `pandas` library is not installed. Please install it to use this function.")
    
    datasets = to_list(
        datasets,
        ["d2t-openweather", "d2t-football", "d2t-gsmarena"]
    )
    nlg_models = to_list(
        nlg_models,
        ["gemma2", "gpt4o", "llama3-3", "phi3-5"]
    )
    splits = to_list(
        splits,
        ["test", "iaa"]
    )
    out_dict = {}

    model_outputs = {
        (model, dataset, split): load_cached_or_download(
            f"https://raw.githubusercontent.com/llm-span-annotators/span-annotation/refs/heads/main/outputs/d2t-eval/{model}/{dataset}-{model}-{split}.jsonl",
            filetype="jsonl",
        )
        for model in nlg_models
        for dataset in datasets
        for split in splits
    }
    model_inputs = {
        (dataset, split): load_cached_or_download(
            f"https://raw.githubusercontent.com/llm-span-annotators/span-annotation/refs/heads/main/inputs/d2t-eval/{dataset}/{split}.json",
            filetype="json",
        )
        for dataset in datasets
        for split in splits
    }

    annotations = {
        split: load_cached_or_download(
            url=f"https://raw.githubusercontent.com/llm-span-annotators/span-annotation/refs/heads/main/annotations/human/d2t-eval/{split}/annotations.jsonl",
            filetype="jsonl",
        )
        for split in splits
    }
    for split in splits:
        out_dict[split] = {}
        df = annotations[split]
        for dataset in datasets:
            out_dict[split][dataset] = {}
            for model in nlg_models:
                print(f"Loading {model} annotations for {dataset}/{split}...")
                filter_df = df[(df["dataset"] == dataset) & (df["setup_id"] == model)]

                all_spans = []
                all_infos = []
                outputs_local = model_outputs[(model, dataset, split)]
                inputs_local = model_inputs[(dataset, split)]
                for _, row in filter_df.iterrows():
                    spans = []
                    infos = {
                        "line_id": row["example_idx"],
                        "tgt": outputs_local[row["example_idx"] == outputs_local["example_idx"]]["output"].iloc[0],
                        # stringify the input to always be the same type
                        "src": json.dumps(inputs_local.iloc[row["example_idx"]].to_dict()),
                        "split": split,
                        "dataset": dataset,
                        "model": model,
                    }
                    for span in row["annotations"]:
                        start_i, end_i = span["start"], span["start"] + len(span["text"])
                        if isinstance(start_i, int) and isinstance(end_i, int) and start_i < end_i:
                            spans.append(
                                Span(
                                    start=start_i,
                                    end=end_i,
                                    label=span["type"],
                                    text=span["text"],
                                )
                            )
                    all_spans.append(spans)
                    all_infos.append(infos)

                if len(all_infos) <= 1:
                    continue
                labl_dataset = LabeledDataset.from_spans(
                    texts=[x["tgt"] for x in all_infos],
                    spans=all_spans,
                    infos=all_infos,
                    tokenizer=tokenizer,
                    tokenizer_kwargs=tokenizer_kwargs,
                )
                out_dict[split][dataset][model] = labl_dataset

    return out_dict