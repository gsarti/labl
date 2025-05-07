import json
from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from typing import cast

from tqdm import tqdm

from labl.data import EditedEntry, LabeledEntry, MultiEditEntry
from labl.data.edited_dataset import EditedDataset
from labl.data.labeled_dataset import LabeledDataset
from labl.datasets import load_divemt, load_qe4pe, load_wmt24esa
from labl.utils.tokenizer import Tokenizer, get_tokenizer


def get_labeled_dataset(
    dataset_name: str,
    langs: str | list[str],
) -> Generator[LabeledDataset | EditedDataset]:
    if isinstance(langs, str):
        langs = [langs]
    if dataset_name == "qe4pe":
        for lang in langs:
            tokenizer = get_tokenizer(
                "facebook/nllb-200-3.3B",
                tokenizer_kwargs={
                    "tgt_lang": "ita_Latn" if lang == "ita" else "nld_Latn",
                    "add_special_tokens": True,
                },
            )
            qe4pe = load_qe4pe(
                langs=lang,  # type: ignore
                configs=["main", "oracle_pe"],
                highlight_modalities="no_highlight",
                tokenizer=tokenizer,
                with_gaps=False,
            )
            main = cast(list[MultiEditEntry], qe4pe["main"][lang])
            oracle_pe = cast(list[MultiEditEntry], qe4pe["oracle_pe"][lang])
            qe4pe_entries = EditedDataset([m + o for m, o in zip(main, oracle_pe, strict=True)])
            yield qe4pe_entries
    elif dataset_name == "divemt":
        lang_code_map = {
            "ara": "ar_AR",
            "ita": "it_IT",
            "nld": "nl_XX",
            "tur": "tr_TR",
            "ukr": "uk_UA",
            "vie": "vi_VN",
        }
        for lang in langs:
            tokenizer = get_tokenizer(
                "facebook/mbart-large-50-one-to-many-mmt",
                tokenizer_kwargs={
                    "tgt_lang": lang_code_map[lang],
                    "add_special_tokens": True,
                },
            )
            dataset = load_divemt(
                configs="main",
                langs=langs,  # type: ignore
                mt_models="mbart50",
                tokenizer=tokenizer,
                with_gaps=False,
            )
            yield dataset["main"][lang]["mbart50"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_src_mt_texts(
    dataset_name: str, langs: str | list[str] | None = None
) -> Generator[tuple[list[str], list[str], str]]:
    if dataset_name == "qe4pe":
        dataset = load_qe4pe(configs="main", langs=langs)  # type: ignore
        for lang in dataset["main"].keys():
            data = dataset["main"][lang]
            source_texts = cast(list[str], [cast(MultiEditEntry, e)[0].info["src_text"] for e in data])
            mt_texts = [cast(MultiEditEntry, e)[0].orig.text for e in data]
            yield source_texts, mt_texts, lang
    elif dataset_name == "divemt":
        dataset = load_divemt(configs="main", langs=langs, mt_models="mbart50")  # type: ignore
        for lang in dataset["main"].keys():
            data = dataset["main"][lang]["mbart50"]
            source_texts = cast(list[str], [e.info["src_text"] for e in data])
            mt_texts = [cast(EditedEntry, e).orig.text for e in data]
            yield source_texts, mt_texts, lang
    elif dataset_name == "wmt24esa":
        dataset = load_wmt24esa(langs=langs, mt_models="Aya23")  # type: ignore
        for lang in dataset["Aya23"].keys():
            data = dataset["Aya23"][lang]
            source_texts = cast(list[str], [e.info["src"] for e in data])
            mt_texts = [cast(LabeledEntry, e).text for e in data]
            yield source_texts, mt_texts, lang
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def build_metrics_dataset(
    unsup_metrics_fname: str | Path,
    sup_metrics_fnames: list[str] | list[Path],
    tokenizer: Tokenizer | None = None,
    reverse_sign_match: list[str] = ["logprob", "mcd_logprob_mean"],
    reverse_sign_startswith: list[str] = ["logit_lens_logprob_layer"],
) -> dict[str, LabeledDataset]:
    """Builds a dictionary of metrics datasets from the given filenames"""
    # Build metrics labeled datasets
    with open(unsup_metrics_fname) as f:
        data = json.load(f)["data"]
    metrics_datasets: dict[str, LabeledDataset] = {}
    unsupervised_metrics: list[str] = [k for k in data[0].keys() if k not in ["src", "mt", "mt_tokens"]]
    for metric in tqdm(unsupervised_metrics, desc="Loading unsup. metrics", total=len(unsupervised_metrics)):
        metrics_datasets[metric] = LabeledDataset.from_tokens(
            texts=[entry["mt"] for entry in data],
            tokens=[entry["mt_tokens"] for entry in data],
            labels=[entry[metric] for entry in data],
            tokenizer=tokenizer,
            show_progress=False,
        )
        if metric in reverse_sign_match or metric.startswith(tuple(reverse_sign_startswith)):
            metrics_datasets[metric].relabel(lambda label: -label if isinstance(label, float | int) else label)
    for fname in sup_metrics_fnames:
        with open(fname) as f:
            data = json.load(f)["data"]
        base_fname = Path(fname).stem if isinstance(fname, Path) else fname.split("/")[-1].split(".")[0]
        metric_name = base_fname.replace("{dataset_name}_", "", 1).replace("_{lang}.json", "", 1)
        entry_dataset = LabeledDataset.from_spans(
            texts=[entry["mt"] for entry in data],
            spans=[entry["error_spans"] for entry in data],
            tokenizer=tokenizer,
            show_progress=False,
        )
        entry_dataset_binary = deepcopy(entry_dataset)
        entry_dataset_binary.relabel(lambda label: 0 if label is None else 1)
        metrics_datasets[f"{metric_name}_binary"] = entry_dataset_binary
        entry_dataset.relabel(lambda label: 0 if label is None else 1 if label == "minor" else 2)
        metrics_datasets[f"{metric_name}_severity"] = entry_dataset
        entry_dataset_confidence = LabeledDataset.from_spans(
            texts=[entry["mt"] for entry in data],
            spans=[
                [
                    {
                        "start": s["start"],
                        "end": s["end"],
                        "label": s["confidence"] * (2 if s["label"] == "major" else 1),
                    }
                    for s in entry["error_spans"]
                ]
                for entry in data
            ],
            tokenizer=tokenizer,
            show_progress=False,
        )
        entry_dataset_confidence.relabel(lambda label: 0.0 if label is None else label)
        metrics_datasets[f"{metric_name}_confidence"] = entry_dataset_confidence
    return metrics_datasets
