"""Classes for tokenizing and detokenizing text."""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import cast

from jiwer import AbstractTransform, Compose, ReduceToListOfListOfWords, Strip
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from labl.utils.aggregation import label_sum_aggregation
from labl.utils.token import LabelType
from labl.utils.transform import SPLIT_REGEX, ReduceToListOfListOfTokens, RegexReduceToListOfListOfWords


class Tokenizer(ABC):
    """Base class for tokenizers.

    This class provides a common interface for tokenizing and detokenizing text, unifying the behavior of
    `jiwer` and `transformers` tokenizers for alignment and visualization.

    Attributes:
        transform (jiwer.AbstractTransform | jiwer.Compose): The transformation to apply to the input strings.
            This should be a composition of transformations that includes a final step producing a list of list of
            tokens, following [jiwer transformations](https://jitsi.github.io/jiwer/reference/transformations/).
        has_bos_token (bool): Whether the tokenizer sets a beginning-of-sequence token. Defaults to False.
        has_eos_token (bool): Whether the tokenizer sets an end-of-sequence token. Defaults to False.
    """

    def __init__(
        self, transform: AbstractTransform | Compose, has_bos_token: bool = False, has_eos_token: bool = False
    ):
        self.transform = transform
        self.has_bos_token = has_bos_token
        self.has_eos_token = has_eos_token

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
        self, texts: str | list[str], add_gaps: bool = False, gap_token: str = "▁"
    ) -> tuple[list[list[str]], list[list[tuple[int, int] | None]]]:
        """Tokenizes the input texts and returns the character spans of the tokens.

        Args:
            texts (str | list[str]): The texts to tokenize.
            add_gaps (bool): Whether gaps should be added before/after tokens and offsets.
            gap_token (str): The token to use for gaps. Default: `▁`.

        Returns:
            The tokens of the input texts, and tuples `(start_idx, end_idx)` marking the position of tokens
            in the original text. If the token is not present in the original text, None is used instead.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _add_gaps_to_tokens_and_offsets(
        self,
        tokens: list[list[str]],
        offsets: list[list[tuple[int, int] | None]],
        gap_token: str = "▁",
    ) -> tuple[list[list[str]], list[list[tuple[int, int] | None]]]:
        """Adds gaps to one or more sequences of tokens and their offsets.

        This is useful for adding annotations of insertions and deletions to the text and edits, respectively.
        The resulting sequence will have 2N + 1 `tokens`, with `gap_token` at even indices like:
        `▁ Hello ▁ World ▁ ! ▁`. The `offsets` will be `None` for the gaps.

        Args:
            tokens (list[list[str]]): One or more lists of tokens to add gaps to.
            offsets (list[list[tuple[int, int]  |  None]]): One or more offsets for the original `tokens`.
            gap_token (str): The token to use for gaps. Default: `▁`.

        Returns:
            A tuple containing a list of list of tokens with gaps and a list of lists with the respective offsets.
        """
        tokens_with_gaps = []
        offsets_with_gaps = []
        for curr_tokens, curr_offsets in zip(tokens, offsets, strict=True):
            curr_tokens_with_gaps = []
            curr_offsets_with_gaps = []
            for idx, (tok, off) in enumerate(zip(curr_tokens, curr_offsets, strict=True)):
                if idx == 0:
                    if not self.has_bos_token:
                        curr_tokens_with_gaps.append(gap_token)
                        curr_offsets_with_gaps.append(None)
                    else:
                        curr_tokens_with_gaps.append(tok)
                        curr_offsets_with_gaps.append(off)
                        continue
                curr_tokens_with_gaps.append(tok)
                curr_offsets_with_gaps.append(off)
                if (idx < len(curr_tokens) - 2 and self.has_eos_token) or not self.has_eos_token:
                    curr_tokens_with_gaps.append(gap_token)
                    curr_offsets_with_gaps.append(None)
            tokens_with_gaps.append(curr_tokens_with_gaps)
            offsets_with_gaps.append(curr_offsets_with_gaps)
        return tokens_with_gaps, offsets_with_gaps

    @staticmethod
    def _merge_gap_annotations(
        all_labels: Sequence[Sequence[LabelType]],
        merge_fn: Callable[[Sequence[LabelType]], LabelType] | None = None,
        has_bos_token: bool = False,
        keep_final_gap: bool = True,
    ) -> Sequence[Sequence[LabelType]]:
        """Merges gap annotations in a list of token_labels.

        Args:
            labels (list[list[str | int | float | None]]): A list containing token annotations that need to be
                merged.
            merge_fn (callable): A callable taking in a sequence of labels (`str | int | float | None`) and returning a
                merged label of type `str | int | float | None`. Default: `None` (add labels together).
            has_bos_token (bool): Whether the token sequence has a beginning-of-sequence token. Default: False.
            keep_final_gap (bool): Whether to keep the final gap token. Default: True.

        Returns:
            A list of tokens with N + 1 tokens, as opposed to 2N + 1 tokens with gaps (assuming no given bos/eos,
            only the last gap is kept to handle end-of-sequence insertions).
        """
        if merge_fn is None:
            merge_fn = label_sum_aggregation
        merged_all_labels = []
        for labels in all_labels:
            merged_labels = []
            gap_label = None
            for idx in range(len(labels)):
                if idx % 2 == 0:  # Even indices are gaps
                    # Final gap is kept regardless of it being an EOS token or not
                    if (has_bos_token and idx == 0) or (idx == len(labels) - 1 and keep_final_gap):
                        merged_labels.append(labels[idx])
                    else:
                        gap_label = labels[idx]
                else:
                    if gap_label is not None and labels[idx] is not None:
                        label = merge_fn([gap_label, labels[idx]])
                    elif labels[idx] is None:
                        label = gap_label
                    else:
                        label = labels[idx]
                    merged_labels.append(label)
                    gap_label = None
            merged_all_labels.append(merged_labels)
        return merged_all_labels

    @staticmethod
    def _remove_gap_tokens(
        all_tokens: list[list[str]],
        has_bos_token: bool,
        keep_final_gap: bool = True,
    ) -> list[list[str]]:
        """Removes gap tokens from a list of tokens.

        Args:
            all_tokens (list[list[str]]): The list of tokens to filter.
            has_bos_token (bool): Whether the token sequence has a beginning-of-sequence token.
            keep_final_gap (bool): Whether to keep the final gap token. Default: True.

        Returns:
            A list of tokens with gaps removed. The final gap token is kept regardless of it being an EOS token
            to handle end-of-sequence insertions.
        """
        out_tokens = []
        for tokens in all_tokens:
            curr_out_tokens = []
            for idx in range(len(tokens)):
                if idx % 2 != 0:  # Even indices are gaps
                    curr_out_tokens.append(tokens[idx])
                    continue
                if (has_bos_token and idx == 0) or (keep_final_gap and idx == len(tokens) - 1):
                    curr_out_tokens.append(tokens[idx])
            out_tokens.append(curr_out_tokens)
        return out_tokens

    @staticmethod
    def _remove_gap_offsets(
        all_offsets: list[list[tuple[int, int] | None]],
        has_bos_token: bool,
        keep_final_gap: bool = True,
    ) -> list[list[tuple[int, int] | None]]:
        """Removes gap offsets from a list of offsets.
        Args:
            all_offsets (list[list[tuple[int, int] | None]]): The list of offsets to filter.
            has_bos_token (bool): Whether the token sequence has a beginning-of-sequence token.
            keep_final_gap (bool): Whether to keep the final gap offset. Default: True.

        Returns:
            A list of offsets with gaps removed. The final gap offset is kept regardless of it being an EOS token
            to handle end-of-sequence insertions.
        """
        return [
            [
                off
                for idx, off in enumerate(offsets)
                if off is not None or (idx == len(offsets) - 1 and keep_final_gap) or (idx == 0 and has_bos_token)
            ]
            for offsets in all_offsets
        ]

    @staticmethod
    def _has_gaps(tokens: list[str], offsets: list[tuple[int, int] | None], gap_token="▁"):
        """Checks if the tokens contain gaps.

        Args:
            tokens (list[list[str]]): The tokens to check.
            offsets (list[list[tuple[int, int] | None]]): The offsets of the tokens.
            gap_token (str): The token used for gaps. Default: `▁`.

        Returns:
            True if gaps are present, False otherwise.
        """
        for idx, (token, offset) in enumerate(zip(tokens, offsets, strict=True)):
            if idx % 2 == 0 and offset is not None:
                return False
            if idx > 0 and idx < len(tokens) - 1 and idx % 2 == 0 and token != gap_token:
                return False
        return True


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

    def _get_offsets(self, tokens: list[str] | list[list[str]]) -> list[list[tuple[int, int] | None]]:
        """Returns the character spans of the tokens in the original text.

        Args:
            tokens (list[str] | list[list[str]]): The tokens of one or more strings.

        Returns:
            A list of lists, each containing a tuple (start_idx, end_idx) marking the token position in the original
            text. If the token is not present in the original text, None is used instead.
        """
        tok_transform: ReduceToListOfListOfWords = self.transform.transforms[-1]  # type: ignore
        delimiter = tok_transform.word_delimiter
        if isinstance(tokens, list) and isinstance(tokens[0], str):
            tokens = cast(list[str], tokens)
            tokens = [tokens]
        all_offsets = []
        for text in tokens:
            text_offsets = []
            start = 0
            for token in text:
                end = start + len(token)
                text_offsets.append((start, end))
                start = end + len(delimiter)
            all_offsets.append(text_offsets)
        assert all(len(t) == len(c) for t, c in zip(tokens, all_offsets, strict=True)), (
            "Token and char span lengths do not match."
        )
        return all_offsets

    def tokenize_with_offsets(
        self, texts: str | list[str], add_gaps: bool = False, gap_token: str = "▁"
    ) -> tuple[list[list[str]], list[list[tuple[int, int] | None]]]:
        """Tokenizes the input texts and returns the character spans of the tokens.

        Args:
            texts (str | list[str]): The strings to tokenize.
            add_gaps (bool): Whether gaps should be added before/after tokens and offsets.
            gap_token (str): The token to use for gaps. Default: `▁`.

        Returns:
            The tokens of the input texts, and tuples (start_idx, end_idx) marking the position of tokens
            in the original text. If the token is not present in the original text, None is used instead.
        """
        tokens = self.transform(texts)
        offsets = self._get_offsets(tokens)
        if add_gaps:
            tokens, offsets = self._add_gaps_to_tokens_and_offsets(tokens, offsets, gap_token=gap_token)
        return tokens, offsets


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

    def tokenize_with_offsets(
        self, texts: str | list[str], add_gaps: bool = False, gap_token: str = "▁"
    ) -> tuple[list[list[str]], list[list[tuple[int, int] | None]]]:
        """Tokenizes the input texts and returns the character spans of the tokens.

        Args:
            texts (str | list[str]): The strings to tokenize.
            add_gaps (bool): Whether gaps should be added before/after tokens and offsets.
            gap_token (str): The token to use for gaps. Default: `▁`.

        Returns:
            The tokens of the input texts, and tuples (start_idx, end_idx) marking the position of tokens
            in the original text. If the token is not present in the original text, None is used instead.
        """
        tok_transform: RegexReduceToListOfListOfWords = self.transform.transforms[-1]  # type: ignore
        expression = tok_transform.exp
        if isinstance(texts, str):
            texts = [texts]
        tokens: list[list[str]] = self.transform(texts)
        all_offsets: list[list[tuple[int, int] | None]] = []
        for text in texts:
            text_offsets = []
            for match in re.finditer(expression, text):
                text_offsets.append(match.span())
            all_offsets.append(text_offsets)
        assert all(len(t) == len(c) for t, c in zip(tokens, all_offsets, strict=True)), (
            "Token and char span lengths do not match."
        )
        if add_gaps:
            tokens, all_offsets = self._add_gaps_to_tokens_and_offsets(tokens, all_offsets, gap_token=gap_token)
        return tokens, all_offsets


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
                **kwargs,
            ),
            has_bos_token=has_bos_token,
            has_eos_token=has_eos_token,
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
        self, texts: str | list[str], add_gaps: bool = False, gap_token: str = "▁"
    ) -> tuple[list[list[str]], list[list[tuple[int, int] | None]]]:
        """Tokenizes the input texts and returns the character spans of the tokens.

        Args:
            texts (str | list[str]): The strings to tokenize.
            add_gaps (bool): Whether gaps should be added before/after tokens and offsets.
            gap_token (str): The token to use for gaps. Default: `▁`.

        Returns:
            The tokens of the input texts, and the character spans of the tokens.

        """
        if not self.transform.tokenizer.is_fast:
            raise RuntimeError("Tokenizer must be a PreTrainedTokenizerFast for char span extraction.")
        if isinstance(texts, str):
            texts = [texts]
        all_tokens = []
        all_offsets = []
        for sentence in texts:
            encoding: BatchEncoding = self.transform.tokenizer(text_target=sentence, return_offsets_mapping=True)
            tokens = encoding.tokens()
            offsets = [tup if tup[0] != 0 or tup[1] != 0 else None for tup in encoding.offset_mapping]
            offsets = cast(list[tuple[int, int] | None], offsets)
            assert len(tokens) == len(offsets), "Token and char span lengths do not match."
            all_tokens.append(tokens)
            all_offsets.append(offsets)
        if add_gaps:
            all_tokens, all_offsets = self._add_gaps_to_tokens_and_offsets(
                all_tokens, all_offsets, gap_token=gap_token
            )
        return all_tokens, all_offsets


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
            "See labl.utils.tokenizer.WhitespaceTokenizer or labl.utils.tokenizer.WordBoundaryTokenizer for examples."
        )
    raise RuntimeError(
        "Invalid tokenizer type. Expected str, Tokenizer or transformers.PreTrainedTokenizer, "
        f"got {type(tokenizer).__name__}."
    )
