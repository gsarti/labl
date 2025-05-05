from collections.abc import Generator
from typing import cast

from labl.data import EditedEntry, LabeledEntry, MultiEditEntry
from labl.datasets import load_divemt, load_qe4pe, load_wmt24esa


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
