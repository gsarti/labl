"""Classes for tokenizing and detokenizing text."""

import re
from logging import getLogger
from typing import cast

from jiwer import ReduceToListOfListOfWords
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

SPLIT_REGEX = r"[\w']+|[.,!?:;'”#$%&\(\)\*\+-/<=>@\[\]^_`{|}~\"]"
logger = getLogger(__name__)


class RegexReduceToListOfListOfWords(ReduceToListOfListOfWords):
    """Version of `ReduceToListOfWords` using Regex for splitting.

    Args:
        exp (str): The Regex expression to use for splitting.
            Defaults to `r"[\\w']+|[.,!?:;'”#$%&\\(\\)\\*\\+-/<=>@\\[\\]^_{|}~\"]`.
            This regex keeps words (including contractions) together as single tokens,
            and treats each punctuation mark or special character as its own separate token.
    """

    def __init__(self, exp: str = SPLIT_REGEX):
        """
        Args:
            exp: the Regex expression to use for splitting."
        """
        self.exp = exp

    def process_string(self, s: str):
        return [[m.group(0) for m in re.finditer(self.exp, s) if len(m.group(0)) >= 1]]


class ReduceToListOfListOfTokens(ReduceToListOfListOfWords):
    """Version of `ReduceToListOfWords` using a tokenizer from the `transformers` library.

    Args:
        tokenizer_or_id (str | PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer or its ID.
            If a string is provided, it will be used to load the tokenizer from the `transformers` library.
        add_special_tokens (bool): Whether to add special tokens to the tokenized output. Defaults to False.
    """

    def __init__(
        self,
        tokenizer_or_id: str | PreTrainedTokenizer | PreTrainedTokenizerFast,
        add_special_tokens: bool = False,
        **kwargs,
    ):
        self.add_special_tokens = add_special_tokens
        if isinstance(tokenizer_or_id, str):
            self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
                tokenizer_or_id, use_fast=True, **kwargs
            )
        else:
            if kwargs:
                logger.warning(f"Ignoring additional keyword arguments for tokenizer initialization: {kwargs}.")
            self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = tokenizer_or_id

    def process_string(self, s: str):
        if self.add_special_tokens:
            ids: list[int] = self.tokenizer(text_target=s).input_ids
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            tokens = cast(list[str], tokens)
            return [tokens]
        tokens = self.tokenizer.tokenize(s)
        return [tokens]
