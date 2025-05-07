# Adapted from https://github.com/wmt-conference/wmt-collect-translations/blob/main/tools/prompts.py

from typing import cast

from transformers.tokenization_utils import PreTrainedTokenizer

from labl.utils.cache import load_cached_or_download

LANG_MAP = {
    "en": "English",
    "cs": "Czech",
    "uk": "Ukrainian",
    "ja": "Japanese",
    "ru": "Russian",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
    "hi": "Hindi",
    "is": "Icelandic",
}
WMT24_TEMPLATE = "Translate the following segment surrounded in triple backlashes into {tgt_lang}. The {src_lang} segment: \n```{text}```\n"
FEW_SHOT_URL = "https://raw.githubusercontent.com/wmt-conference/wmt-collect-translations/refs/heads/main/few_shots/shots.{src_lang}-{tgt_lang}.json"


def load_shots(src_lang: str, tgt_lang: str) -> list[dict]:
    df = load_cached_or_download(FEW_SHOT_URL.format(src_lang=src_lang, tgt_lang=tgt_lang), filetype="json")
    return df.to_dict(orient="records")


def get_chat_prompt_with_few_shots(text: str, lang_pair: str, template: str = WMT24_TEMPLATE) -> list[dict[str, str]]:
    src_lang, tgt_lang = lang_pair.split("-")
    few_shots = load_shots(src_lang, tgt_lang)
    prompt = []
    for shot in few_shots:
        shot_query = template.format(src_lang=LANG_MAP[src_lang], tgt_lang=LANG_MAP[tgt_lang], text=shot["source"])
        prompt.append({"role": "user", "content": shot_query})
        prompt.append({"role": "assistant", "content": f"```{shot['target']}```"})
    example_to_translate = template.format(src_lang=LANG_MAP[src_lang], tgt_lang=LANG_MAP[tgt_lang], text=text)
    prompt.append({"role": "user", "content": example_to_translate})
    return prompt


def get_formatted_source_target_texts(
    src: str, mt: str, lang: str, tokenizer: PreTrainedTokenizer, is_encoder_decoder: bool
) -> tuple[str, str]:
    if is_encoder_decoder:
        return src, mt
    prompt = get_chat_prompt_with_few_shots(src, lang)
    src_updated = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    src_updated = cast(str, src_updated) + "```"
    mt_updated = tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content": f"```{mt}"}], tokenize=False)
    mt_updated = cast(str, mt_updated).rstrip("\n")
    return src_updated, mt_updated
