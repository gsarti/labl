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
    model_id: str = "facebook/nllb-200-3.3B"
    tokenizer_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "src_lang": "eng_Latn",
            "tgt_lang": "ita_Latn",
        }
    )
    dataset_name: Literal["qe4pe", "divemt", "wmt24esa"] = "qe4pe"
    langs: str | list[str] | None = "ita"
    output_dir: str = "outputs"


def main(cfg: Config) -> None:
    model = load_model(cfg.model_id, "dummy", tokenizer_kwargs=cfg.tokenizer_kwargs)
    model: AttributionModel = cast(AttributionModel, torch.compile(model))
    register_step_function(unsupervised_qe_metrics_fn, "unsupervised_qe_metrics_fn", overwrite=True)  # type: ignore
    for src_texts, mt_texts, lang in tqdm(get_src_mt_texts(cfg.dataset_name, langs=cfg.langs)):
        print(f"Processing {lang} ({len(src_texts)} entries)...")
        out_dicts = []
        curr_fname = Path(cfg.output_dir) / f"{cfg.dataset_name}_unsupervised_metrics_{lang}.json"
        curr_fname.parent.mkdir(parents=True, exist_ok=True)
        for src, mt in zip(src_texts, mt_texts, strict=True):
            # Compute metrics
            out = model.attribute(src, mt, step_scores=["unsupervised_qe_metrics_fn"])[0]
            mt_tokens = [t.token for t in out.target[1:]]
            out_metrics = out.step_scores["unsupervised_qe_metrics_fn"]  # type: ignore
            metric_names = get_metric_names(model)
            assert len(metric_names) == out_metrics.shape[1], (
                f"Expected {len(metric_names)} metrics, but got {out_metrics.shape[1]} instead."
            )

            # Add attention metrics
            out_attn = model.attribute(src, mt, method="attention", attribute_target=model.is_encoder_decoder)[0]
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
    cfg = Config()
    main(cfg)
