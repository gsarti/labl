import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch
from inseq import load_model, register_step_function
from inseq.models import HuggingfaceModel
from torch.distributions import Categorical
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer

from unsup_wqe import get_metric_names, get_src_mt_texts, unsupervised_qe_metrics_fn
from unsup_wqe.model_utils import get_attributions, get_mt_tokens_and_metrics
from unsup_wqe.prompt_utils import get_formatted_source_target_texts


@dataclass
class Config:
    model_id: str
    dataset_name: Literal["qe4pe", "divemt", "wmt24esa"]
    langs: str | list[str] | None
    output_dir: str = "outputs/metrics/{dataset_name}"
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        cfg = cls(
            model_id=args.model_id,
            dataset_name=args.dataset_name,
            langs=args.langs,
            output_dir=args.output_dir,
        )
        if args.src_lang is not None and args.tgt_lang is not None:
            cfg.tokenizer_kwargs = {
                "src_lang": args.src_lang,
                "tgt_lang": args.tgt_lang,
            }
        return cfg


def main(cfg: Config) -> None:
    print("A")
    model = load_model(
        cfg.model_id,
        "dummy",
        tokenizer_kwargs=cfg.tokenizer_kwargs,
        model_kwargs={
            "attn_implementation": "eager",
            "load_in_4bit": True,        
            "bnb_4bit_compute_dtype": torch.float16,
            "device_map": "auto",
        }
    )  # type: ignore
    
    print("B")
    model: HuggingfaceModel = cast(HuggingfaceModel, model)
    print("Model in eval mode:", not model.model.training)
    print("C")
    register_step_function(unsupervised_qe_metrics_fn, "unsupervised_qe_metrics_fn", overwrite=True)  # type: ignore
    print("D")
    for src_texts, mt_texts, lang in tqdm(get_src_mt_texts(cfg.dataset_name, langs=cfg.langs)):
        out_dicts = []
        if "{dataset_name}" in cfg.output_dir:
            cfg.output_dir = cfg.output_dir.format(dataset_name=cfg.dataset_name)
        curr_fname = Path(cfg.output_dir) / f"{cfg.dataset_name}_unsupervised_metrics_{lang}.json"
        curr_fname.parent.mkdir(parents=True, exist_ok=True)
        if curr_fname.exists():
            with open(curr_fname) as f:
                out_dicts = json.load(f)["data"]
        start_idx = len(out_dicts)
        if start_idx > 0:
            print(f"Skipping {len(out_dicts)} entries already processed for {lang}...")
        sources = src_texts[start_idx:]
        targets = mt_texts[start_idx:]
        print(f"Processing {lang} ({len(sources)} entries)...")
        for src, mt in tqdm(zip(sources, targets, strict=True), desc="Processing entries", total=len(sources)):
            # Compute metrics
            fmt_src, fmt_mt = get_formatted_source_target_texts(
                src, mt, lang, cast(PreTrainedTokenizer, model.tokenizer), model.is_encoder_decoder
            )
            out = model.attribute(fmt_src, fmt_mt, step_scores=["unsupervised_qe_metrics_fn"], show_progress=False)[0]
            mt_tokens, out_metrics = get_mt_tokens_and_metrics(out, model)
            metric_names = get_metric_names(model)
            assert len(metric_names) == out_metrics.shape[1], (
                f"Expected {len(metric_names)} metrics, but got {out_metrics.shape[1]} instead."
            )

            # Add attention metrics
            out_attn = model.attribute(
                fmt_src, fmt_mt, method="attention", attribute_target=model.is_encoder_decoder, show_progress=False
            )[0]
            attn_scores = get_attributions(out_attn)
            attn_entropy_mean = []
            attn_entropy_max = []
            for gen_step in attn_scores:
                distr = (
                    gen_step[~torch.isnan(gen_step)].reshape(-1, gen_step.shape[1], gen_step.shape[2]).permute(1, 2, 0)
                )
                attn_entropy = Categorical(probs=distr).entropy().flatten()
                attn_entropy_mean.append(attn_entropy.mean().item())
                attn_entropy_max.append(attn_entropy.max().item())
            attn_entropy_mean = torch.tensor(attn_entropy_mean).unsqueeze(1)
            attn_entropy_max = torch.tensor(attn_entropy_max).unsqueeze(1)
            all_metrics = torch.cat([out_metrics, attn_entropy_mean, attn_entropy_max], dim=1).T
            metric_names += ["attn_entropy_mean", "attn_entropy_max"]
            out_curr = {
                "src": src,
                "mt": mt,
                "mt_tokens": mt_tokens,
            }
            for metric_name, metric_values in zip(metric_names, all_metrics, strict=False):
                assert len(metric_values) == len(mt_tokens), (
                    f"Expected {len(mt_tokens)} values, but got {len(metric_values)} metrics instead.\n"
                    f"Metric name: {metric_name}\nTokens: {mt_tokens}\n"
                )
                out_curr[metric_name] = metric_values.tolist()
            out_dicts.append(out_curr)
            with open(curr_fname, "w") as f:
                json.dump({"data": out_dicts}, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute unsupervised metrics")
    parser.add_argument("--model_id", type=str, help="Model ID")
    parser.add_argument("--dataset_name", type=str, choices=["qe4pe", "divemt", "wmt24esa"], help="Dataset name")
    parser.add_argument("--langs", type=str, nargs="+", help="Languages to process")
    parser.add_argument("--src_lang", type=str, help="Source language", default=None)
    parser.add_argument("--tgt_lang", type=str, help="Target language", default=None)
    parser.add_argument("--output_dir", type=str, help="Output directory", default="outputs")
    args = parser.parse_args()
    cfg = Config.from_args(args)
    main(cfg)
