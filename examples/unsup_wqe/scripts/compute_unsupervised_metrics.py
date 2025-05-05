import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch
from inseq import AttributionModel, load_model, register_step_function
from torch.distributions import Categorical
from tqdm import tqdm

from unsup_wqe import get_metric_names, get_src_mt_texts, unsupervised_qe_metrics_fn


@dataclass
class Config:
    model_id: str
    dataset_name: Literal["qe4pe", "divemt", "wmt24esa"]
    langs: str | list[str] | None
    output_dir: str = "outputs"
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
    model = load_model(
        cfg.model_id, "dummy", tokenizer_kwargs=cfg.tokenizer_kwargs, model_kwargs={"attn_implementation": "eager"}
    )
    model: AttributionModel = cast(AttributionModel, torch.compile(model))
    register_step_function(unsupervised_qe_metrics_fn, "unsupervised_qe_metrics_fn", overwrite=True)  # type: ignore
    for src_texts, mt_texts, lang in tqdm(get_src_mt_texts(cfg.dataset_name, langs=cfg.langs)):
        out_dicts = []
        curr_fname = Path(cfg.output_dir) / f"{cfg.dataset_name}_unsupervised_metrics_{lang}.json"
        curr_fname.parent.mkdir(parents=True, exist_ok=True)
        if curr_fname.exists():
            with open(curr_fname) as f:
                out_dicts = json.load(f)["data"]
        start_idx = len(out_dicts)
        sources = src_texts[start_idx:]
        targets = mt_texts[start_idx:]
        print(f"Processing {lang} ({len(sources)} entries)...")
        for src, mt in tqdm(zip(sources, targets, strict=True), desc="Processing entries", total=len(sources)):
            # Compute metrics
            out = model.attribute(src, mt, step_scores=["unsupervised_qe_metrics_fn"], show_progress=False)[0]
            mt_tokens = [t.token for t in out.target[1:]]
            out_metrics = out.step_scores["unsupervised_qe_metrics_fn"]  # type: ignore
            metric_names = get_metric_names(model)
            assert len(metric_names) == out_metrics.shape[1], (
                f"Expected {len(metric_names)} metrics, but got {out_metrics.shape[1]} instead."
            )

            # Add attention metrics
            out_attn = model.attribute(
                src, mt, method="attention", attribute_target=model.is_encoder_decoder, show_progress=False
            )[0]
            if (
                model.is_encoder_decoder
                and isinstance(out_attn.source_attributions, torch.Tensor)
                and isinstance(out_attn.target_attributions, torch.Tensor)
            ):
                attn_scores = (
                    torch.cat([out_attn.source_attributions, out_attn.target_attributions], dim=0).permute(1, 0, 2, 3)
                    / 2
                )
            elif out_attn.target_attributions is not None:
                attn_scores = out_attn.target_attributions.permute(1, 0, 2, 3)
            else:
                raise ValueError("No attention scores found in the output.")
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
