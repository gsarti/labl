import argparse
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from sklearn.metrics import auc, average_precision_score, precision_recall_curve

from labl.data.base_sequence import BaseMultiLabelEntry
from labl.data.edited_dataset import EditedDataset
from labl.data.edited_entry import EditedEntry, MultiEditEntry
from labl.data.labeled_dataset import LabeledDataset
from labl.data.labeled_entry import MultiLabelEntry
from labl.utils.tokenizer import get_tokenizer
from unsup_wqe.data_utils import build_metrics_dataset, get_labeled_dataset


@dataclass
class Config:
    dataset_name: Literal["qe4pe", "divemt", "wmt24esa"]
    langs: list[str]
    tokenizer_name: str
    lang_codes: list[str] | None = None
    outputs_dir: str = "outputs/results/{dataset_name}"
    output_fname: str = "{dataset_name}_metrics_performance.json"
    metrics_dir: str = "outputs/metrics/{dataset_name}"
    unsup_metrics_fname: str = "{dataset_name}_unsupervised_metrics_{lang}.json"
    sup_metrics_fnames: list[str] = field(
        default_factory=lambda: [
            # "{dataset_name}_xcomet_lite_{lang}.json", # Omitted since it does not output word-level error spans
            "{dataset_name}_xcomet_xl_{lang}.json",
            # "{dataset_name}_xcomet_xxl_{lang}.json",
        ]
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        cfg = cls(
            dataset_name=args.dataset_name,
            langs=args.langs,
            tokenizer_name=args.tokenizer_name,
            lang_codes=args.lang_codes,
            outputs_dir=args.outputs_dir,
            metrics_dir=args.metrics_dir,
            output_fname=args.output_fname,
            unsup_metrics_fname=args.unsup_metrics_fname,
            sup_metrics_fnames=args.sup_metrics_fnames,
        )
        return cfg


def main(cfg: Config) -> None:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    out_metric_scores = {}
    for idx, (lang, data) in enumerate(
        zip(cfg.langs, get_labeled_dataset(cfg.dataset_name, langs=cfg.langs), strict=True)
    ):
        out_metric_scores[lang] = {}
        tokenizer_kwargs: dict[str, Any] = {"add_special_tokens": True}
        if cfg.lang_codes is not None:
            tokenizer_kwargs["tgt_lang"] = cfg.lang_codes[idx]
        tokenizer = get_tokenizer(cfg.tokenizer_name, tokenizer_kwargs=tokenizer_kwargs)
        if "{dataset_name}" in cfg.outputs_dir:
            cfg.outputs_dir = cfg.outputs_dir.format(dataset_name=cfg.dataset_name)
        if "{dataset_name}" in cfg.metrics_dir:
            cfg.metrics_dir = cfg.metrics_dir.format(dataset_name=cfg.dataset_name)
        unsup_metrics_fname = Path(cfg.metrics_dir) / cfg.unsup_metrics_fname.format(
            dataset_name=cfg.dataset_name, lang=lang
        )
        sup_metric_fnames = [
            Path(cfg.metrics_dir) / fname.format(dataset_name=cfg.dataset_name, lang=lang)
            for fname in cfg.sup_metrics_fnames
        ]
        metrics_datasets = build_metrics_dataset(
            unsup_metrics_fname=unsup_metrics_fname, sup_metrics_fnames=sup_metric_fnames, tokenizer=tokenizer
        )
        for metric, metric_data in metrics_datasets.items():
            if isinstance(data[0], BaseMultiLabelEntry):
                out_metric_scores[lang][metric] = {}
                entry = cast(BaseMultiLabelEntry, data[0])
                num_annotators = len(entry)
                for ann_idx in range(num_annotators):
                    if isinstance(entry, MultiEditEntry):
                        curr_annotator_data = LabeledDataset([cast(MultiEditEntry, e)[ann_idx].orig for e in data])
                    elif isinstance(entry, MultiLabelEntry):
                        curr_annotator_data = LabeledDataset([cast(MultiLabelEntry, e)[ann_idx] for e in data])
                    else:
                        raise TypeError(f"Unsupported entry type: {type(entry)}")
                    copy_annotator_data = deepcopy(curr_annotator_data)
                    copy_annotator_data.relabel(lambda lab: 1.0 if lab is not None else 0.0)
                    try:
                        labels_array = metric_data._get_labels_array(float, copy_annotator_data)
                        metric_array, annotator_array = labels_array[0], labels_array[1]
                        precision, recall, _ = precision_recall_curve(annotator_array, metric_array)
                        auprc = auc(recall, precision)
                        ap = average_precision_score(annotator_array, metric_array)
                    except (RuntimeError, ValueError) as e:
                        raise RuntimeError(f"Error computing scores for {metric}") from e
                    out_metric_scores[lang][metric][ann_idx] = {"ap": ap, "auprc": auprc}
            else:
                if isinstance(data, EditedDataset):
                    curr_annotator_data = LabeledDataset([cast(EditedEntry, e).orig for e in data])
                else:
                    curr_annotator_data = cast(LabeledDataset, data)
                copy_annotator_data = deepcopy(curr_annotator_data)
                copy_annotator_data.relabel(lambda lab: 1.0 if lab is not None else 0.0)
                try:
                    labels_array = metric_data._get_labels_array(float, copy_annotator_data)
                    metric_array, annotator_array = labels_array[0], labels_array[1]
                    precision, recall, _ = precision_recall_curve(annotator_array, metric_array)
                    auprc = auc(recall, precision)
                    ap = average_precision_score(annotator_array, metric_array)
                except (RuntimeError, ValueError) as e:
                    raise RuntimeError(f"Error computing scores for {metric}") from e
                out_metric_scores[lang][metric] = {"ap": ap, "auprc": auprc}
        with open(Path(cfg.outputs_dir) / cfg.output_fname.format(dataset_name=cfg.dataset_name), "w") as f:
            json.dump(out_metric_scores, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute unsupervised metrics")
    parser.add_argument("--dataset_name", type=str, choices=["qe4pe", "divemt", "wmt24esa"], help="Dataset name")
    parser.add_argument("--langs", type=str, nargs="+", help="Languages to process")
    parser.add_argument("--tokenizer_name", type=str, help="Tokenizer name")
    parser.add_argument(
        "--lang_codes",
        type=str,
        nargs="+",
        help="Language codes for the tokenizer (optional, default: None)",
        default=None,
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        help="Output directory for the results",
        default="outputs",
    )
    parser.add_argument(
        "--output_fname",
        type=str,
        help="Output filename for the results",
        default="{dataset_name}_metrics_performance.json",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        help="Directory for the metrics files",
        default="outputs/metrics/{dataset_name}",
    )
    parser.add_argument(
        "--unsup_metrics_fname",
        type=str,
        help="Filename for the unsupervised metrics",
        default="{dataset_name}_unsupervised_metrics_{lang}.json",
    )
    parser.add_argument(
        "--sup_metrics_fnames",
        type=str,
        nargs="+",
        help="Filenames for the supervised metrics",
        default=[
            # "{dataset_name}_xcomet_lite_{lang}.json", # Omitted since it does not output word-level error spans
            "{dataset_name}_xcomet_xl_{lang}.json",
            # "{dataset_name}_xcomet_xxl_{lang}.json",
        ],
    )
    args = parser.parse_args()
    cfg = Config.from_args(args)
    main(cfg)
