import json
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Literal, cast

from tqdm import tqdm

from labl.data import EditedDataset, LabeledEntry, MultiEditEntry
from labl.data.labeled_dataset import LabeledDataset
from labl.datasets import load_qe4pe
from labl.utils.tokenizer import get_tokenizer
from unsup_wqe.data_utils import build_metrics_dataset


@dataclass
class Config:
    outputs_dir: str = "outputs/results/qe4pe"
    metrics_dir: str = "outputs/metrics/qe4pe"
    unsup_metrics_fname: str = "qe4pe_unsupervised_metrics_{lang}.json"
    sup_metrics_fnames: list[str] = field(
        default_factory=lambda: [
            # "qe4pe_xcomet_lite_{lang}.json", # Omitted since it does not output word-level error spans
            "qe4pe_xcomet_xl_{lang}.json",
            "qe4pe_xcomet_xxl_{lang}.json",
            "qe4pe_xcomet_xl_cont_{lang}.json",
            "qe4pe_xcomet_xxl_cont_{lang}.json",
        ]
    )
    output_fname: str = "qe4pe_metrics_correlations.json"
    langs: dict[Literal["ita", "nld"], str] = field(default_factory=lambda: {"ita": "ita_Latn", "nld": "nld_Latn"})


def main(cfg: Config) -> None:
    out_correlations = {}
    for lang, lang_code in cfg.langs.items():
        out_correlations[lang] = {}
        tokenizer = get_tokenizer(
            "facebook/nllb-200-3.3B",
            tokenizer_kwargs={
                "tgt_lang": lang_code,
                "add_special_tokens": True,
            },
        )
        unsup_metrics_fname = Path(cfg.metrics_dir) / cfg.unsup_metrics_fname.format(lang=lang)
        sup_metric_fnames = [Path(cfg.metrics_dir) / fname.format(lang=lang) for fname in cfg.sup_metrics_fnames]
        metrics_datasets = build_metrics_dataset(
            unsup_metrics_fname=unsup_metrics_fname, sup_metrics_fnames=sup_metric_fnames, tokenizer=tokenizer
        )

        # Load all no_highlight edits (3 in main, 3 in oracle_pe)
        qe4pe = load_qe4pe(
            langs=lang,
            configs=["main", "oracle_pe"],
            highlight_modalities="no_highlight",
            tokenizer=tokenizer,
            with_gaps=False,
        )
        main = cast(list[MultiEditEntry], qe4pe["main"][lang])
        oracle_pe = cast(list[MultiEditEntry], qe4pe["oracle_pe"][lang])
        all_entries = EditedDataset([m + o for m, o in zip(main, oracle_pe, strict=True)])

        num_annotators = len(cast(MultiEditEntry, all_entries[0]))
        annotators_ids = list(range(num_annotators))
        annotator_combinations = []
        for i in range(1, len(annotators_ids) + 1):
            annotator_combinations.extend(combinations(annotators_ids, i))
        for metric in metrics_datasets.keys():
            out_correlations[lang][metric] = {}
        out_correlations[lang]["human_baseline"] = {}

        for combination in tqdm(
            annotator_combinations, desc="Building correlation dict", total=len(annotator_combinations)
        ):
            curr_num_annotators = len(combination)
            unselected_num_annotators = num_annotators - curr_num_annotators
            # Build edit count labeled dataset
            edit_count_entries = []
            # Unselected annotators are used to compute the human baseline performance
            unselected_annotators_entries: list[MultiEditEntry] = []
            for idx in range(len(all_entries)):
                multi_edit_entry = cast(MultiEditEntry, all_entries[idx])

                # Get only the selected annotators
                selected_multi_edit_entry = MultiEditEntry([multi_edit_entry[comb_idx] for comb_idx in combination])
                unselected_multi_edit_entry = MultiEditEntry(
                    [
                        multi_edit_entry[comb_idx]
                        for comb_idx in range(len(multi_edit_entry))
                        if comb_idx not in combination
                    ]
                )
                edit_count_entry = LabeledEntry.from_tokens(
                    tokens=selected_multi_edit_entry[0].orig.tokens,
                    labels=selected_multi_edit_entry.label_counts,
                    text=selected_multi_edit_entry[0].orig.text,
                    offsets=selected_multi_edit_entry[0].orig.tokens_offsets,
                )
                edit_count_entries.append(edit_count_entry)
                unselected_annotators_entries.append(unselected_multi_edit_entry)
            edit_count_entries = LabeledDataset(edit_count_entries)
            for metric, metric_data in metrics_datasets.items():
                try:
                    score = metric_data.get_correlation(edit_count_entries).score
                except (RuntimeError, ValueError) as e:
                    raise RuntimeError(
                        f"Error computing correlation for {metric} with {curr_num_annotators} annotators"
                    ) from e
                if curr_num_annotators in out_correlations[lang][metric]:
                    out_correlations[lang][metric][curr_num_annotators] += [score]
                else:
                    out_correlations[lang][metric][curr_num_annotators] = [score]
            for idx in range(unselected_num_annotators):
                curr_annotator_entries = LabeledDataset(deepcopy([e[idx].orig for e in unselected_annotators_entries]))
                curr_annotator_entries.relabel(lambda label: 0 if label is None else 1)
                try:
                    score = curr_annotator_entries.get_correlation(edit_count_entries).score
                except (RuntimeError, ValueError) as e:
                    raise RuntimeError(
                        f"Error computing correlation for annotator {idx} with {curr_num_annotators} annotators"
                    ) from e
                if curr_num_annotators in out_correlations[lang]["human_baseline"]:
                    out_correlations[lang]["human_baseline"][curr_num_annotators] += [score]
                else:
                    out_correlations[lang]["human_baseline"][curr_num_annotators] = [score]
            with open(Path(cfg.outputs_dir) / cfg.output_fname, "w") as f:
                json.dump(out_correlations, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
