import json
from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from typing import cast

from tqdm import tqdm

from labl.data import EditedEntry, LabeledEntry, MultiEditEntry, MultiLabelEntry
from labl.data.base_sequence import BaseMultiLabelEntry
from labl.data.edited_dataset import EditedDataset
from labl.data.labeled_dataset import LabeledDataset
from labl.datasets import load_divemt, load_qe4pe, load_wmt24esa
from labl.utils.tokenizer import Tokenizer, get_tokenizer


def get_labl_tokenizer(dataset_name: str, lang: str) -> Tokenizer:
    if dataset_name == "qe4pe":
        return get_tokenizer(
            "facebook/nllb-200-3.3B",
            tokenizer_kwargs={
                "tgt_lang": "ita_Latn" if lang == "ita" else "nld_Latn",
                "add_special_tokens": True,
            },
        )
    elif dataset_name == "divemt":
        lang_code_map = {
            "ara": "ar_AR",
            "ita": "it_IT",
            "nld": "nl_XX",
            "tur": "tr_TR",
            "ukr": "uk_UA",
            "vie": "vi_VN",
        }
        return get_tokenizer(
            "facebook/mbart-large-50-one-to-many-mmt",
            tokenizer_kwargs={
                "tgt_lang": lang_code_map[lang],
                "add_special_tokens": True,
            },
        )
    elif dataset_name == "wmt24esa":
        return get_tokenizer("CohereLabs/aya-23-35B", tokenizer_kwargs={"add_special_tokens": True})
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_labeled_dataset(
    dataset_name: str,
    langs: str | list[str],
) -> Generator[LabeledDataset | EditedDataset]:
    if isinstance(langs, str):
        langs = [langs]
    if dataset_name == "qe4pe":
        for lang in langs:
            tokenizer = get_labl_tokenizer(dataset_name, lang)
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
        for lang in langs:
            tokenizer = get_labl_tokenizer(dataset_name, lang)
            dataset = load_divemt(
                configs="main",
                langs=lang,  # type: ignore
                mt_models="mbart50",
                tokenizer=tokenizer,
                with_gaps=False,
            )
            yield dataset["main"][lang]["mbart50"]
    elif dataset_name == "wmt24esa":
        for lang in langs:
            tokenizer = get_labl_tokenizer(dataset_name, lang)
            dataset = load_wmt24esa(
                langs=lang,  # type: ignore
                mt_models="Aya23",
                tokenizer=tokenizer,
            )
            yield dataset["Aya23"][lang]
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
    unsup_metrics_fname: str | Path | None,
    sup_metrics_fnames: list[str] | list[Path],
    dataset_name: str | None = None,
    lang: str | None = None,
    tokenizer: Tokenizer | None = None,
    reverse_sign_match: list[str] = ["logprob", "mcd_logprob_mean"],
    reverse_sign_startswith: list[str] = ["logit_lens_logprob_layer"],
) -> dict[str, LabeledDataset]:
    """Builds a dictionary of metrics datasets from the given filenames"""
    # Build metrics labeled datasets
    metrics_datasets: dict[str, LabeledDataset] = {}
    if unsup_metrics_fname is not None:
        with open(unsup_metrics_fname) as f:
            data = json.load(f)["data"]
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
        metric_name = Path(fname).stem if isinstance(fname, Path) else fname.split("/")[-1].split(".")[0]
        if dataset_name is not None:
            metric_name = metric_name.replace(f"{dataset_name}_", "", 1)
        if lang is not None:
            metric_name = metric_name.replace(f"_{lang}", "", 1)
        # Binary metric
        if "error_spans" in data[0]:
            entry_dataset = LabeledDataset.from_spans(
                texts=[entry["mt"] for entry in data],
                spans=[entry["error_spans"] for entry in data],
                tokenizer=tokenizer,
                show_progress=False,
            )
            entry_dataset_binary = deepcopy(entry_dataset)
            entry_dataset_binary.relabel(lambda label: 0 if label is None else 1)
            metrics_datasets[metric_name] = entry_dataset_binary
        # Continuous metric
        else:
            if "xcomet_xl_cont" in metric_name:
                xcomet_tokenizer = get_tokenizer("facebook/xlm-roberta-xl")
            elif "xcomet_xxl_cont" in metric_name:
                xcomet_tokenizer = get_tokenizer("facebook/xlm-roberta-xxl")
            else:
                raise ValueError(f"Unknown continuous metric: {metric_name}")
            entry_dataset = LabeledDataset.from_tokens(
                texts=[entry["mt"] for entry in data],
                tokens=[entry["mt_tokens"][1:-1] for entry in data],
                labels=[
                    [
                        min + maj + crit
                        for min, maj, crit in zip(
                            entry["minor_error_probs"][1:-1],
                            entry["major_error_probs"][1:-1],
                            entry["critical_error_probs"][1:-1],
                            strict=True,
                        )
                    ]
                    for entry in data
                ],
                tokenizer=xcomet_tokenizer,
            )
            entry_dataset.retokenize(tokenizer, label_aggregation_fn=lambda labels: labels[0])
            entry_dataset.relabel(lambda label: 0.0 if label is None else label)
            metrics_datasets[metric_name] = entry_dataset

    return metrics_datasets


def get_binary_annotator_data(
    data: LabeledDataset | EditedDataset, annotator_idx: int | None = None
) -> LabeledDataset:
    entry = data[0]
    if isinstance(entry, EditedEntry):
        annotator_data = LabeledDataset([cast(EditedEntry, e).orig for e in data])
    elif isinstance(entry, LabeledEntry):
        annotator_data = cast(LabeledDataset, data)
    elif isinstance(entry, MultiEditEntry) and annotator_idx is not None:
        annotator_data = LabeledDataset([cast(MultiEditEntry, e)[annotator_idx].orig for e in data])
    elif isinstance(entry, MultiLabelEntry) and annotator_idx is not None:
        annotator_data = LabeledDataset([cast(MultiLabelEntry, e)[annotator_idx] for e in data])
    else:
        raise TypeError(f"Unsupported entry type: {type(entry)}")
    copy_annotator_data = deepcopy(annotator_data)
    copy_annotator_data.relabel(lambda lab: 1.0 if lab is not None else 0.0)
    return copy_annotator_data


def get_num_annotators(data: LabeledDataset | EditedDataset) -> int:
    if isinstance(data[0], BaseMultiLabelEntry):
        entry = cast(BaseMultiLabelEntry, data[0])
        return len(entry)
    return 1
