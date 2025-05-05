from typing import Any, Literal

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils.import_utils import is_pandas_available

from labl.data.labeled_dataset import LabeledDataset
from labl.utils.cache import load_cached_or_download
from labl.utils.span import Span
from labl.utils.tokenizer import Tokenizer
from labl.utils.typing import to_list

Wmt24EsaLanguage = Literal["en-cs", "en-ja", "en-es", "en-zh", "en-hi", "en-is", "cs-uk", "en-uk", "en-ru"]
Wmt24EsaDomain = Literal["speech", "social", "news", "literary", "education", "voice", "personal", "official"]
Wmt24EsaMTModel = Literal[
    "Unbabel-Tower70B",
    "CUNI-GA",
    "Gemini-1.5-Pro",
    "SCIR-MT",
    "Aya23",
    "Claude-3.5",
    "ONLINE-W",
    "Llama3-70B",
    "GPT-4",
    "CommandR-plus",
    "IKUN-C",
    "refA",
    "IOL-Research",
    "CUNI-DocTransformer",
    "IKUN",
    "CUNI-MH",
    "Mistral-Large",
    "ONLINE-B",
    "Dubformer",
    "MSLC",
    "Team-J",
    "HW-TSC",
    "NTTSU",
    "TranssionMT",
    "AMI",
    "CUNI-Transformer",
    "ONLINE-G",
    "Yandex",
]


def load_wmt24esa(
    langs: Wmt24EsaLanguage | list[Wmt24EsaLanguage] | None = None,
    domains: Wmt24EsaDomain | list[Wmt24EsaDomain] | None = None,
    mt_models: Wmt24EsaMTModel | list[Wmt24EsaMTModel] | None = None,
    tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
    tokenizer_kwargs: dict[str, Any] = {},
) -> dict[str, dict[str, LabeledDataset]]:
    """Load the WMT24 ESA annotations from [Kocmi et al. (2024)](https://aclanthology.org/2024.wmt-1.1/), containing
        partially overlapping segments across multiple language pairs with a single set of
        [ESA annotations](https://aclanthology.org/2024.wmt-1.131/) over multiple MT system outputs.

    Args:
        langs (Wmt24EsaLanguage | list[Wmt24EsaLanguage] | None):
            One or more languages to load. Defaults to `["en-cs", "en-ja", "en-es", "en-zh", "en-hi", "en-is", "cs-uk",
                "en-uk", "en-ru"]. Available options: `"en-cs", "en-ja", "en-es", "en-zh", "en-hi", "en-is", "cs-uk",
                "en-uk", "en-ru"`.
        domains (Wmt24EsaDomain | list[Wmt24EsaDomain] | None):
            One or more text categories to load. Defaults to `["speech", "social", "news", "literary", "education",
                "voice", "personal", "official"]`. Available options: `"speech", "social", "news", "literary",
                "education", "voice", "personal", "official"`.
        mt_models (Wmt24EsaMTModel | list[Wmt24EsaMTModel] | None):
            One or more models for which annotations need to be loaded. Defaults to all models.
            Available options: `"Unbabel-Tower70B", "CUNI-GA", "Gemini-1.5-Pro", "SCIR-MT", "Aya23", "Claude-3.5",
                "ONLINE-W", "Llama3-70B", "GPT-4", "CommandR-plus", "IKUN-C", "refA", "IOL-Research",
                "CUNI-DocTransformer", "IKUN", "CUNI-MH", "Mistral-Large", "ONLINE-B", "Dubformer", "MSLC",
                "Team-J", "HW-TSC", "NTTSU", "TranssionMT", "AMI", "CUNI-Transformer", "ONLINE-G", "Yandex"`.
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

    langs = to_list(langs, ["en-cs", "en-ja", "en-es", "en-zh", "en-hi", "en-is", "cs-uk", "en-uk", "en-ru"])
    domains = to_list(domains, ["speech", "social", "news", "literary", "education", "voice", "personal", "official"])
    mt_models = to_list(
        mt_models,
        [
            "Unbabel-Tower70B",
            "CUNI-GA",
            "Gemini-1.5-Pro",
            "SCIR-MT",
            "Aya23",
            "Claude-3.5",
            "ONLINE-W",
            "Llama3-70B",
            "GPT-4",
            "CommandR-plus",
            "IKUN-C",
            "refA",
            "IOL-Research",
            "CUNI-DocTransformer",
            "IKUN",
            "CUNI-MH",
            "Mistral-Large",
            "ONLINE-B",
            "Dubformer",
            "MSLC",
            "Team-J",
            "HW-TSC",
            "NTTSU",
            "TranssionMT",
            "AMI",
            "CUNI-Transformer",
            "ONLINE-G",
            "Yandex",
        ],
    )
    out_dict = {}
    df = load_cached_or_download(
        url="https://raw.githubusercontent.com/wmt-conference/wmt24-news-systems/refs/heads/main/jsonl/wmt24_esa.jsonl",
        filetype="jsonl",
    )
    for model in mt_models:
        out_dict[model] = {}
        for lang in langs:
            filter_df = df[df["domain"].isin(domains) & (df["system"] == model) & (df["langs"] == lang)]
            all_spans = []
            all_infos = []
            if not filter_df.empty:
                print(f"Loading {model} annotations for {lang}...")
                for _, row in filter_df.iterrows():
                    spans = []
                    infos = {c: row[c] for c in ["line_id", "doc_id", "domain", "esa_score", "annotator", "src"]}
                    for span in row["esa_spans"]:
                        start, end = span["start_i"], span["end_i"]
                        if isinstance(start, int) and isinstance(end, int) and start < end:
                            spans.append(
                                Span(
                                    start=span["start_i"],
                                    end=span["end_i"],
                                    label=span["severity"],
                                    text=row["tgt"][start:end],
                                )
                            )
                    all_spans.append(spans)
                    all_infos.append(infos)
                labl_dataset = LabeledDataset.from_spans(
                    texts=list(filter_df["tgt"]),
                    spans=all_spans,
                    infos=all_infos,
                    tokenizer=tokenizer,
                    tokenizer_kwargs=tokenizer_kwargs,
                )
                out_dict[model][lang] = labl_dataset
    return out_dict
