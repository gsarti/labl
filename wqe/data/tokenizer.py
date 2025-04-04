"""Classes for tokenizing and detokenizing text."""

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast

from jiwer import AbstractTransform, Compose, ReduceToListOfListOfWords, Strip
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from wqe.data.transform import SPLIT_REGEX, ReduceToListOfListOfTokens, RegexReduceToListOfListOfWords


class Tokenizer(ABC):
    """Base class for tokenizers.

    This class provides a common interface for tokenizing and detokenizing text, unifying the behavior of
    `jiwer` and `transformers` tokenizers for alignment and visualization.

    Attributes:
        transform (jiwer.AbstractTransform | jiwer.Compose): The transformation to apply to the input strings.
            This should be a composition of transformations that includes a final step producing a list of list of
            tokens, following [jiwer transformations](https://jitsi.github.io/jiwer/reference/transformations/).
    """

    def __init__(self, transform: AbstractTransform | Compose):
        self.transform = transform

    def __call__(
        self, texts: str | list[str], with_offsets: bool = False
    ) -> list[list[str]] | tuple[list[list[str]], Sequence[Sequence[tuple[int, int] | None]]]:
        """Tokenizes one or more input strings.

        Args:
            texts (str | list[str]): The strings to tokenize.
            with_offsets (bool): If True, returns the (start, end) character indices of the tokens.
                If False, returns only the tokens.

        Returns:
            The tokens of the input strings, and optionally the character spans of the tokens.

        """
        return self.tokenize(texts) if not with_offsets else self.tokenize_with_offsets(texts)

    def tokenize(self, texts: str | list[str]) -> list[list[str]]:
        """Tokenizes one or more input texts.

        Args:
            texts (str | list[str]): The strings to tokenize.

        Returns:
            A list of lists, each containing the tokens of the corresponding input string.
        """
        return self.transform(texts)

    @abstractmethod
    def detokenize(self, tokens: list[str] | list[list[str]]) -> list[str]:
        """Detokenizes the input tokens.

        Args:
            tokens (list[str] | list[list[str]]): The tokens of one or more strings to detokenize.

        Returns:
            A list containing the detokenized string(s).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def tokenize_with_offsets(
        self, texts: str | list[str]
    ) -> tuple[list[list[str]], Sequence[Sequence[tuple[int, int] | None]]]:
        """Tokenizes the input texts and returns the character spans of the tokens.

        Args:
            texts (str | list[str]): The texts to tokenize.

        Returns:
            The tokens of the input texts, and tuples (start_idx, end_idx) marking the position of tokens
            in the original text. If the token is not present in the original text, None is used instead.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class WhitespaceTokenizer(Tokenizer):
    """Tokenizer that uses whitespace to split the input strings into tokens.

    Hardcodes the `Compose([Strip(), ReduceToListOfListOfWords()])` transformation for tokenization.

    Args:
        word_delimiter (str): The delimiter to use for splitting words. Defaults to whitespace.
    """

    def __init__(self, word_delimiter: str = " "):
        super().__init__(transform=Compose([Strip(), ReduceToListOfListOfWords(word_delimiter=word_delimiter)]))

    def detokenize(self, tokens: list[str] | list[list[str]]) -> list[str]:
        """Detokenizes the input tokens using whitespace.

        Args:
            tokens (list[str] | list[list[str]]): The tokens of one or more strings to detokenize.

        Returns:
            A list containing the detokenized string(s).
        """
        tok_transform: ReduceToListOfListOfWords = self.transform.transforms[-1]  # type: ignore
        if isinstance(tokens, list) and isinstance(tokens[0], str):
            tokens = cast(list[str], tokens)
            tokens = [tokens]
        return [tok_transform.word_delimiter.join(sentence) for sentence in tokens]

    def tokenize_with_offsets(
        self, texts: str | list[str]
    ) -> tuple[list[list[str]], list[list[tuple[int, int] | None]]]:
        """Tokenizes the input texts and returns the character spans of the tokens.

        Args:
            texts (str | list[str]): The strings to tokenize.

        Returns:
            The tokens of the input texts, and tuples (start_idx, end_idx) marking the position of tokens
            in the original text. If the token is not present in the original text, None is used instead.
        """
        tok_transform: ReduceToListOfListOfWords = self.transform.transforms[-1]  # type: ignore
        delimiter = tok_transform.word_delimiter
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.transform(texts)
        tokens = cast(list[list[str]], tokens)
        char_spans = []
        for sentence in tokens:
            sentence_spans = []
            start = 0
            for token in sentence:
                end = start + len(token)
                sentence_spans.append((start, end))
                start = end + len(delimiter)
            char_spans.append(sentence_spans)
        assert all(len(t) == len(c) for t, c in zip(tokens, char_spans, strict=True)), (
            "Token and char span lengths do not match."
        )
        return tokens, char_spans


class WordBoundaryTokenizer(Tokenizer):
    """Tokenizer that uses word boundaries to split the input strings into tokens.

    Hardcodes the `Compose([Strip(), RegexReduceToListOfListOfWords()])` transformation for tokenization.

    Args:
        exp (str): The Regex expression to use for splitting.
            Defaults to `r"[\\w']+|[.,!?:;'”#$%&\\(\\)\\*\\+-/<=>@\\[\\]^_{|}~\"]`.
            This regex keeps words (including contractions) together as single tokens,
            and treats each punctuation mark or special character as its own separate token.
    """

    def __init__(self, exp: str = SPLIT_REGEX):
        super().__init__(transform=Compose([Strip(), RegexReduceToListOfListOfWords(exp=exp)]))

    def _detokenize_str(self, tokens: list[str]) -> str:
        result = ""
        for i, token in enumerate(tokens):
            if i == 0:
                result = token
                continue
            if len(token) == 1 and token in ".,!?:;')\"]}%&*+/<=>#@`|~":
                result += token
            elif i > 0 and tokens[i - 1] in "([{\"'$#":
                result += token
            else:
                result += " " + token
        return result

    def detokenize(self, tokens: list[str] | list[list[str]]) -> list[str]:
        """Detokenizes the input tokens using word boundaries.

        Args:
            tokens (list[str] | list[list[str]]): The tokens of one or more texts to detokenize.

        Returns:
            A list containing the detokenized string(s).
        """
        if isinstance(tokens, list) and isinstance(tokens[0], str):
            tokens = cast(list[str], tokens)
            tokens = [tokens]
        tokens = cast(list[list[str]], tokens)
        return [self._detokenize_str(sentence) for sentence in tokens]

    def tokenize_with_offsets(self, texts: str | list[str]) -> tuple[list[list[str]], list[list[tuple[int, int]]]]:
        """Tokenizes the input texts and returns the character spans of the tokens.

        Args:
            texts (str | list[str]): The texts to tokenize.

        Returns:
            The tokens of the input texts, and tuples (start_idx, end_idx) marking the position of tokens
            in the original text. If the token is not present in the original text, None is used instead.
        """
        tok_transform: RegexReduceToListOfListOfWords = self.transform.transforms[-1]  # type: ignore
        expression = tok_transform.exp
        if isinstance(texts, str):
            texts = [texts]
        tokens: list[list[str]] = self.transform(texts)
        char_spans: list[list[tuple[int, int]]] = []
        for sentence in texts:
            sentence_spans = []
            for match in re.finditer(expression, sentence):
                sentence_spans.append(match.span())
            char_spans.append(sentence_spans)
        assert all(len(t) == len(c) for t, c in zip(tokens, char_spans, strict=True)), (
            "Token and char span lengths do not match."
        )
        return tokens, char_spans


class HuggingfaceTokenizer(Tokenizer):
    """Tokenizer that uses a `transformers.PreTrainedTokenizer` to split the input strings into tokens.
    Hardcodes the `ReduceToListOfListOfTokens` transformation for tokenization.

    Args:
        tokenizer_or_id (str | PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer or its ID.
            If a string is provided, it will be used to load the tokenizer from the `transformers` library.
        add_special_tokens (bool): Whether to add special tokens to the tokenized output. Defaults to False.
        has_bos_token (bool): Whether the tokenizer sets a beginning-of-sequence token. Defaults to True.
        has_eos_token (bool): Whether the tokenizer sets an end-of-sequence token. Defaults to True.
        kwargs (dict): Additional keyword arguments to pass to the tokenizer initialization.
    """

    def __init__(
        self,
        tokenizer_or_id: str | PreTrainedTokenizer | PreTrainedTokenizerFast,
        add_special_tokens: bool = False,
        has_bos_token: bool = True,
        has_eos_token: bool = True,
        **kwargs,
    ):
        super().__init__(
            transform=ReduceToListOfListOfTokens(
                tokenizer_or_id,
                add_special_tokens=add_special_tokens,
                has_bos_token=has_bos_token,
                has_eos_token=has_eos_token,
                **kwargs,
            )
        )
        self.transform = cast(ReduceToListOfListOfTokens, self.transform)

    def detokenize(self, tokens: list[str] | list[list[str]]) -> list[str]:
        if isinstance(tokens, list) and isinstance(tokens[0], str):
            tokens = cast(list[str], tokens)
            ids = self.transform.tokenizer.convert_tokens_to_ids(tokens)
            return [self.transform.tokenizer.decode(ids, skip_special_tokens=True)]
        tokens = cast(list[list[str]], tokens)
        return [
            self.transform.tokenizer.decode(
                self.transform.tokenizer.convert_tokens_to_ids(sentence), skip_special_tokens=True
            )
            for sentence in tokens
        ]

    def tokenize_with_offsets(
        self, texts: str | list[str]
    ) -> tuple[list[list[str]], list[list[tuple[int, int] | None]]]:
        """Tokenizes the input texts and returns the character spans of the tokens.

        Args:
            texts (str | list[str]): The texts to tokenize.

        Returns:
            The tokens of the input texts, and the character spans of the tokens.

        """
        if not self.transform.tokenizer.is_fast:
            raise RuntimeError("Tokenizer must be a PreTrainedTokenizerFast for char span extraction.")
        if isinstance(texts, str):
            texts = [texts]
        all_tokens = []
        all_char_spans = []
        for sentence in texts:
            encoding: BatchEncoding = self.transform.tokenizer(text_target=sentence, return_offsets_mapping=True)
            tokens = encoding.tokens()
            char_spans = [tup if tup[0] != 0 or tup[1] != 0 else None for tup in encoding.offset_mapping]
            char_spans = cast(list[tuple[int, int] | None], char_spans)
            assert len(tokens) == len(char_spans), "Token and char span lengths do not match."
            all_tokens.append(tokens)
            all_char_spans.append(char_spans)
        return all_tokens, all_char_spans


def get_tokenizer(
    tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
    tokenizer_kwargs: dict = {},
) -> Tokenizer:
    if tokenizer is None:
        return WhitespaceTokenizer()
    if isinstance(tokenizer, Tokenizer):
        return tokenizer
    if isinstance(tokenizer, str | PreTrainedTokenizer | PreTrainedTokenizerFast):
        return HuggingfaceTokenizer(tokenizer, **tokenizer_kwargs)
    if isinstance(tokenizer, AbstractTransform | Compose):
        raise RuntimeError(
            "Jiwer transform are supported by defining classes specifying an additional decoding method."
            "See wqe.data.tokenizer.WhitespaceTokenizer or wqe.data.tokenizer.WordBoundaryTokenizer for examples."
        )
    raise RuntimeError(
        "Invalid tokenizer type. Expected str, Tokenizer or transformers.PreTrainedTokenizer, "
        f"got {type(tokenizer).__name__}."
    )
