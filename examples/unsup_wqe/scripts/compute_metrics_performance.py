import argparse
import json
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from labl.data.labeled_dataset import LabeledDataset
from unsup_wqe.data_utils import (
    build_metrics_dataset,
    get_binary_annotator_data,
    get_labeled_dataset,
    get_labl_tokenizer,
    get_num_annotators,
)


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
    sup_metrics_fnames: list[str] = field(default_factory=lambda: [])
    num_random_baselines: int = 10

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
            num_random_baselines=args.num_random_baselines,
        )
        return cfg


def get_metric_scores(ref: LabeledDataset, pred: LabeledDataset, label_type: type) -> dict[str, Any]:
    """Compute metric scores for two datasets containing reference and predicted labels."""
    labels_array = pred._get_labels_array(float, ref)
    pred_array, ref_array = labels_array[0], labels_array[1]
    precisions, recalls, thresholds = precision_recall_curve(ref_array, pred_array)
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    auprc = auc(recalls, precisions)
    ap = average_precision_score(ref_array, pred_array)
    if label_type is float:
        f1_scores = [
            (2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i]))
            if precisions[i] + recalls[i] != 0
            else 0.0
            for i in range(len(thresholds))
        ]
        max_f1_idx = np.argmax(f1_scores)
        f1 = f1_scores[max_f1_idx]
        precision = precisions[max_f1_idx]
        recall = recalls[max_f1_idx]
    elif label_type is int:
        f1 = f1_score(ref_array, pred_array)
        precision = precision_score(ref_array, pred_array)
        recall = recall_score(ref_array, pred_array)
    else:
        raise ValueError(f"Unsupported label type: {label_type}")
    return {"ap": ap, "auprc": auprc, "precision": precision, "recall": recall, "f1": f1}


def compute_scores_fn(metric_scores: dict[int, dict[str, float]], fn: Callable | None = None) -> dict[str, float]:
    vals = list(metric_scores.keys())
    if fn is None:
        fn = lambda scores: sum(scores) / len(scores)
    return {
        metric_name: fn([metric_scores[v][metric_name] for v in vals]) for metric_name in metric_scores[vals[0]].keys()
    }


def get_random_array(data: LabeledDataset) -> LabeledDataset:
    random_data = deepcopy(data)
    random_data.relabel(lambda lab: np.random.randn())
    return random_data


def get_random_baseline_results(cfg: Config, annotator_data: LabeledDataset) -> dict[str, float]:
    out = {}
    for baseline_idx in range(cfg.num_random_baselines):
        baseline_data = get_random_array(annotator_data)
        try:
            metrics_dict = get_metric_scores(annotator_data, baseline_data, baseline_data.label_types[0])
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Error computing scores for baseline {baseline_idx}") from e
        out[baseline_idx] = metrics_dict
    return compute_scores_fn(out)


def main(cfg: Config) -> None:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    out_metric_scores = {}
    if "{dataset_name}" in cfg.outputs_dir:
        cfg.outputs_dir = cfg.outputs_dir.format(dataset_name=cfg.dataset_name)
    Path(cfg.outputs_dir).mkdir(parents=True, exist_ok=True)
    if "{dataset_name}" in cfg.metrics_dir:
        cfg.metrics_dir = cfg.metrics_dir.format(dataset_name=cfg.dataset_name)
    num_annotators = None
    for lang, data in zip(cfg.langs, get_labeled_dataset(cfg.dataset_name, langs=cfg.langs), strict=True):
        out_metric_scores[lang] = {}
        num_annotators = get_num_annotators(data)
        tokenizer = get_labl_tokenizer(cfg.dataset_name, lang)
        unsup_metrics_fname = Path(cfg.metrics_dir) / cfg.unsup_metrics_fname.format(
            dataset_name=cfg.dataset_name, lang=lang
        )
        sup_metric_fnames = [
            Path(cfg.metrics_dir) / fname.format(dataset_name=cfg.dataset_name, lang=lang)
            for fname in cfg.sup_metrics_fnames
        ]
        metrics_datasets = build_metrics_dataset(
            unsup_metrics_fname=unsup_metrics_fname,
            sup_metrics_fnames=sup_metric_fnames,
            dataset_name=cfg.dataset_name,
            lang=lang,
            tokenizer=tokenizer,
        )
        if num_annotators > 1:
            for metric, metric_data in metrics_datasets.items():
                out_metric_scores[lang][metric] = {}
                for ann_idx in range(num_annotators):
                    annotator_data = get_binary_annotator_data(data, ann_idx)
                    try:
                        metrics_dict = get_metric_scores(annotator_data, metric_data, metric_data.label_types[0])
                    except (RuntimeError, ValueError) as e:
                        raise RuntimeError(f"Error computing scores for {metric}") from e
                    out_metric_scores[lang][metric][ann_idx] = metrics_dict
                mean_scores = compute_scores_fn(out_metric_scores[lang][metric])
                out_metric_scores[lang][metric]["mean"] = mean_scores
            out_metric_scores[lang]["random_baseline"] = {}
            out_metric_scores[lang]["human_annotators"] = {}
            for ann_idx in range(num_annotators):
                annotator_data = get_binary_annotator_data(data, ann_idx)
                random_baseline_results = get_random_baseline_results(cfg, annotator_data)
                out_metric_scores[lang]["random_baseline"][ann_idx] = random_baseline_results
                other_ann_idxs = [i for i in range(num_annotators) if i != ann_idx]
                out_metric_scores[lang]["human_annotators"][ann_idx] = {}
                for other_idx in other_ann_idxs:
                    other_annotator_data = get_binary_annotator_data(data, other_idx)
                    try:
                        metrics_dict = get_metric_scores(
                            annotator_data, other_annotator_data, other_annotator_data.label_types[0]
                        )
                    except (RuntimeError, ValueError) as e:
                        raise RuntimeError(f"Error computing scores for annotator pair {ann_idx}, {other_idx}") from e
                    out_metric_scores[lang]["human_annotators"][ann_idx][other_idx] = metrics_dict
                curr_ann_mean_scores = compute_scores_fn(out_metric_scores[lang]["human_annotators"][ann_idx])
                out_metric_scores[lang]["human_annotators"][ann_idx]["mean"] = curr_ann_mean_scores
                curr_ann_min_scores = compute_scores_fn(
                    out_metric_scores[lang]["human_annotators"][ann_idx], lambda x: min(x)
                )
                out_metric_scores[lang]["human_annotators"][ann_idx]["min"] = curr_ann_min_scores
                curr_ann_max_scores = compute_scores_fn(
                    out_metric_scores[lang]["human_annotators"][ann_idx], lambda x: max(x)
                )
                out_metric_scores[lang]["human_annotators"][ann_idx]["max"] = curr_ann_max_scores
            baseline_mean_scores = compute_scores_fn(out_metric_scores[lang]["random_baseline"])
            out_metric_scores[lang]["random_baseline"]["mean"] = baseline_mean_scores
            annotators_mean_scores = compute_scores_fn(
                {i: out_metric_scores[lang]["human_annotators"][i]["mean"] for i in range(num_annotators)}
            )
            out_metric_scores[lang]["human_annotators"]["mean"] = annotators_mean_scores
            annotators_min_scores = compute_scores_fn(
                {i: out_metric_scores[lang]["human_annotators"][i]["min"] for i in range(num_annotators)},
            )
            out_metric_scores[lang]["human_annotators"]["min"] = annotators_min_scores
            annotators_max_scores = compute_scores_fn(
                {i: out_metric_scores[lang]["human_annotators"][i]["max"] for i in range(num_annotators)},
            )
            out_metric_scores[lang]["human_annotators"]["max"] = annotators_max_scores
        else:
            annotator_data = get_binary_annotator_data(data)
            for metric, metric_data in metrics_datasets.items():
                try:
                    metrics_dict = get_metric_scores(annotator_data, metric_data, metric_data.label_types[0])
                except (RuntimeError, ValueError) as e:
                    raise RuntimeError(f"Error computing scores for {metric}") from e
                out_metric_scores[lang][metric] = metrics_dict
            random_baseline_results = get_random_baseline_results(cfg, annotator_data)
            out_metric_scores[lang]["random_baseline"] = random_baseline_results

        with open(Path(cfg.outputs_dir) / cfg.output_fname.format(dataset_name=cfg.dataset_name), "w") as f:
            json.dump(out_metric_scores, f, indent=4, ensure_ascii=False)
    if num_annotators is not None:
        out_metric_scores["mean"] = {}
        for metric_name in out_metric_scores[cfg.langs[0]].keys():
            out_metric_scores["mean"][metric_name] = compute_scores_fn(
                {
                    idx: out_metric_scores[lang][metric_name]
                    if num_annotators == 1
                    else out_metric_scores[lang][metric_name]["mean"]
                    for idx, lang in enumerate(cfg.langs)
                }
            )
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
        default="outputs/results/{dataset_name}",
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
            "{dataset_name}_xcomet_xxl_{lang}.json",
        ],
    )
    parser.add_argument(
        "--num_random_baselines",
        type=int,
        help="Number of random baselines to compute",
        default=10,
    )
    args = parser.parse_args()
    cfg = Config.from_args(args)
    main(cfg)
