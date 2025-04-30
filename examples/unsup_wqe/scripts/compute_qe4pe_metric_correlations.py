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


@dataclass
class Config:
    outputs_dir: str = "outputs"
    metric_fname: str = "unsupervised_metrics_{lang}.json"
    output_fname: str = "unsupervised_metrics_correlations.json"
    langs: dict[Literal["ita", "nld"], str] = field(default_factory=lambda: {"ita": "ita_Latn", "nld": "nld_Latn"})
    single_metrics: list[str] = field(default_factory=lambda: ["logprobs", "mcd_avg", "mcd_var", "logit_lens_depth"])
    per_layer_metrics: list[str] = field(
        default_factory=lambda: ["blood", "logit_lens", "kl_div", "mcd_kl_div", "mcd_kl_div_var"]
    )
    layer_start: int = 0
    layer_end: int = 23


def main(cfg: Config) -> None:
    out_correlations = {}
    for lang, lang_code in cfg.langs.items():
        out_correlations[lang] = {}
        tokenizer = get_tokenizer(
            "facebook/nllb-200-3.3B",
            tokenizer_kwargs={
                "tgt_lang": lang_code,
                "add_special_tokens": False,
            },
        )

        # Build metrics labeled datasets
        with open(Path(cfg.outputs_dir) / cfg.metric_fname.format(lang=lang)) as f:
            data = json.load(f)
        metrics_datasets = {}
        all_metrics = deepcopy(cfg.single_metrics)
        for metric in cfg.per_layer_metrics:
            all_metrics += [f"{metric}_layer_{idx}" for idx in range(cfg.layer_start, cfg.layer_end + 1)]
        for metric in tqdm(all_metrics, desc="Loading metrics", total=len(all_metrics)):
            out_correlations[lang][metric] = {}
            metrics_datasets[metric] = LabeledDataset.from_tokens(
                texts=[entry["mt"] + "</s>" for entry in data],
                tokens=[entry["mt_tokens"] for entry in data],
                labels=[entry[metric] for entry in data],
                tokenizer=tokenizer,
                show_progress=False,
            )

        # Load all no_highlight edits (3 in main, 3 in oracle_pe)
        qe4pe = load_qe4pe(
            langs=lang,
            configs=["main", "oracle_pe"],
            highlight_modalities="no_highlight",
            tokenizer=tokenizer,
            with_gaps=False,
            keep_final_gap=True,
            gap_token="</s>",  # To match the EOS preserved in the unsupervised metrics
        )
        main = cast(list[MultiEditEntry], qe4pe["main"][lang])
        oracle_pe = cast(list[MultiEditEntry], qe4pe["oracle_pe"][lang])
        all_entries = EditedDataset([m + o for m, o in zip(main, oracle_pe, strict=True)])
        assert all(me[0].orig.tokens == de["mt_tokens"] for (me, de) in zip(main, data, strict=True))

        num_annotators = list(range(len(cast(MultiEditEntry, all_entries[0]))))
        annotator_combinations = []
        for i in range(1, len(num_annotators) + 1):
            annotator_combinations.extend(combinations(num_annotators, i))

        for combination in tqdm(
            annotator_combinations, desc="Building correlation dict", total=len(annotator_combinations)
        ):
            curr_num_annotators = len(combination)
            # Build edit count labeled dataset
            edit_count_entries = []
            for idx in range(len(all_entries)):
                multi_edit_entry = cast(MultiEditEntry, all_entries[idx])

                # Get only the selected annotators
                multi_edit_entry = MultiEditEntry([multi_edit_entry[comb_idx] for comb_idx in combination])
                edit_count_entry = LabeledEntry.from_tokens(
                    tokens=multi_edit_entry[0].orig.tokens,
                    labels=multi_edit_entry.label_counts,
                    text=multi_edit_entry[0].orig.text,
                    offsets=multi_edit_entry[0].orig.tokens_offsets,
                )
                edit_count_entries.append(edit_count_entry)
            edit_count_entries = LabeledDataset(edit_count_entries)
            for metric in all_metrics:
                score = metrics_datasets[metric].get_correlation(edit_count_entries).score
                if curr_num_annotators in out_correlations[lang][metric]:
                    out_correlations[lang][metric][curr_num_annotators] += [score]
                else:
                    out_correlations[lang][metric][curr_num_annotators] = [score]
            with open(Path(cfg.outputs_dir) / cfg.output_fname, "w") as f:
                json.dump(out_correlations, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
