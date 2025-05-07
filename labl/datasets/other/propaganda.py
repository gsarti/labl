from typing import Any, Literal

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils.import_utils import is_pandas_available

from labl.data.labeled_dataset import LabeledDataset
from labl.utils.cache import load_cached_or_download
from labl.utils.span import Span
from labl.utils.tokenizer import Tokenizer


def load_propaganda(
    tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
    tokenizer_kwargs: dict[str, Any] = {},
) -> dict[str, dict[str, LabeledDataset]]:
    """Load the propaganda span annotations from [Kasner et al. (2025)](https://arxiv.org/abs/2504.08697) over multiple NLG system outputs.

    Args:
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
    
    outputs_local = load_cached_or_download(
        f"https://raw.githubusercontent.com/llm-span-annotators/span-annotation/refs/heads/main/outputs/propaganda-techniques/test.jsonl",
        filetype="jsonl",
    )
    filter_df = load_cached_or_download(
        url=f"https://raw.githubusercontent.com/llm-span-annotators/span-annotation/refs/heads/main/annotations/human/propaganda/annotations.jsonl",
        filetype="jsonl",
    )

    print(f"Loading annotations for propaganda...")

    all_spans = []
    all_infos = []
    for _, row in filter_df.iterrows():
        spans = []

        infos = {
            "line_id": row["example_idx"],
            "tgt": outputs_local[row["example_idx"] == outputs_local["example_idx"]]["output"].iloc[0],
            "dataset": "propaganda",
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

    labl_dataset = LabeledDataset.from_spans(
        texts=[x["tgt"] for x in all_infos],
        spans=all_spans,
        infos=all_infos,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
    )

    return labl_dataset