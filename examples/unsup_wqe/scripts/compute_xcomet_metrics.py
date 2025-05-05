import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from comet import download_model, load_from_checkpoint
from tqdm import tqdm

from unsup_wqe import get_src_mt_texts
from unsup_wqe.xcomet_lite import XCOMETLiteMetric


@dataclass
class Config:
    model_id: Literal["myyycroft/XCOMET-lite", "Unbabel/XCOMET-XL", "Unbabel/XCOMET-XXL"]
    dataset_name: Literal["qe4pe", "divemt", "wmt24esa"]
    langs: str | list[str] | None
    output_dir: str = "outputs"
    batch_size: int = 1

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        cfg = cls(
            model_id=args.model_id,
            dataset_name=args.dataset_name,
            langs=args.langs,
            output_dir=args.output_dir,
        )
        return cfg


def get_nick(model_id: str) -> str:
    if model_id == "myyycroft/XCOMET-lite":
        return "lite"
    elif model_id == "Unbabel/XCOMET-XL":
        return "xl"
    elif model_id == "Unbabel/XCOMET-XXL":
        return "xxl"
    else:
        raise ValueError(f"Unknown model ID: {model_id}")


def main(cfg: Config) -> None:
    if cfg.model_id in ["Unbabel/XCOMET-XL", "Unbabel/XCOMET-XXL"]:
        model_path = download_model(cfg.model_id)
        model = load_from_checkpoint(model_path)
    else:
        model = XCOMETLiteMetric.from_pretrained(cfg.model_id)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    nickname = get_nick(cfg.model_id)
    for src_texts, mt_texts, lang in tqdm(get_src_mt_texts(cfg.dataset_name, langs=cfg.langs)):
        out_dicts = []
        curr_fname = Path(cfg.output_dir) / f"{cfg.dataset_name}_xcomet_{nickname}_{lang}.json"
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
        out = model.predict(
            [{"src": src, "mt": mt} for src, mt in zip(sources, targets, strict=True)],
            batch_size=cfg.batch_size,
            progress_bar=True,
            accelerator="auto",
            gpus=0 if not torch.cuda.is_available() else 1,
        )
        for src, mt, sent_score, error_spans in zip(
            sources,
            targets,
            out.scores,  # type: ignore
            out.metadata.error_spans,  # type: ignore
            strict=True,
        ):
            data = {
                "src": src,
                "mt": mt,
                "sent_score": sent_score,
                "error_spans": [
                    {"start": s["start"], "end": s["end"], "label": s["severity"], "confidence": s["confidence"]}
                    for s in error_spans
                ],
            }
            out_dicts.append(data)
        with open(curr_fname, "w") as f:
            json.dump({"data": out_dicts}, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute XCOMET metrics")
    parser.add_argument("--model_id", type=str, help="Model ID for the XCOMET model")
    parser.add_argument("--dataset_name", type=str, choices=["qe4pe", "divemt", "wmt24esa"], help="Dataset name")
    parser.add_argument("--langs", type=str, nargs="+", help="Languages to process")
    parser.add_argument("--output_dir", type=str, help="Output directory", default="outputs")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing", default=1)
    args = parser.parse_args()
    cfg = Config.from_args(args)
    main(cfg)
