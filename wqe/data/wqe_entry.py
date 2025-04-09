import re
from collections.abc import Sequence
from logging import getLogger
from textwrap import dedent, indent
from typing import cast, overload
from warnings import warn

from jiwer import WordOutput
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from wqe.data.aggregation import LabelAggregation, label_sumlen_aggregation
from wqe.data.jiwer_ext import process_words
from wqe.data.mixin import AlignedSequencesMixin
from wqe.data.span import EditSpan, ListOfListsOfSpans, Span, SpanList
from wqe.data.token import LabeledToken, LabeledTokenInput, LabeledTokenList, ListOfListsOfLabeledToken
from wqe.data.tokenizer import Tokenizer, WhitespaceTokenizer, get_tokenizer

logger = getLogger(__name__)


class WQEEntry(AlignedSequencesMixin):
    """Class for a single text entry with word-level quality estimation utilities.

    The class provides methods to extract spans, tokenize the text, and align the text and edits. Regardless
    of the initialization method, the class will always contain the original text with its tagged and tokenized
    versions with labels for each token. If edits are provided, for all text-edit pairs a tagged and tokenized
    version of the text and its edits will be created. All tags are also available in the form of spans.

    Attributes:
        text (str):
            The original text.
        spans (list[Span] | list[list[EditSpan]]): If only `text` is specified, `spans` is a list of
            `Span` items containing information about specific spans in `text`. If one or more `edits` are provided,
            `spans` is a list where the i-th element contains a list `EditSpan` with information about `text` and
            the corresponding i-th edit in `edits`.
        tagged (str | list[str]): Tagged version of `text` containing information from `spans`. If multiple edits
            are provided, `tagged` is a list where the i-th element contains the tagged version of `text` with
            information from the i-th edit in `edits`.
        tokens (list[LabeledToken] | list[list[LabeledToken]]): Tokenized variant of `text` with text or numeric
            labels for each token. If multiple edits are provided, `tokens` is a list where the i-th element
            contains a tokenized version of `text` with text or numeric labels for each token.
        tokens_offsets (list[tuple[int, int] | None]): Offsets for each token in `tokens`. Initialized automatically
            when `tokens` is created. The i-th element corresponds to the i-th token in `tokens`.
        edits (list[str] | None):
            One or more edited version of the text. Set only if the entry was created from one or more edits.
        edits_tagged (list[str] | None): Tagged versions (one per edit) of `edits` containing information from `spans`.
            Set only if the entry was created from one or more edits.
        edits_tokens (list[list[LabeledToken]] | None): List where the i-th element corresponds to a list of tokens for
            the corresponding edit in `edits` with text or numeric labels for each token. Set only if the entry was
            created from one or more edits.
        edits_tokens_offsets (list[list[tuple[int, int] | None]] | None): If edits are provided, for every list of
            `edit_tokens` this contains a list with the offsets for each token.
        aligned (list[WordOutput] | None): If one or more edits are provided, this is a list of aligned WordOutputs
            (one per edit) tokenized using the provided tokenizer. The alignment is done using the `jiwer` library.
        has_bos_token (bool): Whether the token sequence has a beginning-of-sequence token.
        has_eos_token (bool): Whether the token sequence has an end-of-sequence token.
        has_gaps (bool | None): Whether the token sequence has gaps. Gaps are used for text/edit pairs to mark the
            positions of insertions and deletions in the original/edited texts, respectively. This is a bool only if
            the entry was initialized with `.from_edits`, and is `None` otherwise. If `False`, it means gap annotations
            were merged to the next token to the right.
    """

    # Private constructor key to prevent direct instantiation
    __constructor_key = object()

    def __init__(
        self,
        text: str,
        spans: SpanList[Span] | ListOfListsOfSpans[EditSpan],
        tagged: str | list[str],
        tokens: LabeledTokenList | ListOfListsOfLabeledToken,
        tokens_offsets: list[tuple[int, int] | None],
        edits: list[str] | None = None,
        edits_tagged: list[str] | None = None,
        edits_tokens: ListOfListsOfLabeledToken | None = None,
        edits_tokens_offsets: list[list[tuple[int, int] | None]] | None = None,
        aligned: list[WordOutput] | None = None,
        has_bos_token: bool = False,
        has_eos_token: bool = False,
        has_gaps: bool | None = None,
        constructor_key: object | None = None,
    ):
        """Private constructor for `WQEEntry`.

        A `WQEEntry` can be initialized from:

        * A `tagged` text, e.g. `Hello <error>world</error>!`, using `WQEEntry.from_tagged(tagged=...)`.

        * A `text` and one or more `edits`, e.g. `Hello world!` and `["Goodbye world!", "Hello planet!"]`, using
            `WQEEntry.from_edits(text=..., edits=...)`.

        * A `text` and a list of labeled `spans`, e.g. `Hello world!` and `[{'start': 0, 'end': 5, 'label': 'error'}]`,
            using `WQEEntry.from_spans(text=..., spans=...)`.

        * A list of `labeled_tokens` with string/numeric labels, e.g. `[('Hel', 0.5), ('lo', 0.7), ('world', 1),
            ('!', 0)]`, or two separate lists of `tokens` and `labels` using `WQEEntry.from_tokens(labeled_tokens=...)`
            or `WQEEntry.from_tokens(tokens=..., labels=)`.
        """
        if constructor_key != self.__constructor_key:
            raise RuntimeError(
                dedent("""\
                The default constructor for `WQEEntry` is private. A `WQEEntry` can be initialized from:

                * A `tagged` text, e.g. `Hello <error>world</error>!`, using `WQEEntry.from_tagged(tagged=...)`.

                * A `text` and one or more `edits`, e.g. `Hello world!` and `["Goodbye world!", "Hello planet!"]`, using
                    `WQEEntry.from_edits(text=..., edits=...)`.

                * A `text` and a list of labeled `spans`, e.g. `Hello world!` and `[{'start': 0, 'end': 5, 'label': 'error'}]`,
                    using `WQEEntry.from_spans(text=..., spans=...)`.

                * A list of `labeled_tokens` with string/numeric labels, e.g. `[('Hel', 0.5), ('lo', 0.7), ('world', 1),
                    ('!', 0)]`, or two separate lists of `tokens` and `labels` using `WQEEntry.from_tokens(labeled_tokens=...)`
                    or `WQEEntry.from_tokens(tokens=..., labels=)`.
                """)
            )
        self._text = text
        self._spans = spans
        self._tagged = tagged
        self._tokens = tokens
        self._tokens_offsets = tokens_offsets
        self._edits = edits
        self._edits_tagged = edits_tagged
        self._edits_tokens = edits_tokens
        self._edits_tokens_offsets = edits_tokens_offsets
        self._aligned = aligned
        self._has_bos_token = has_bos_token
        self._has_eos_token = has_eos_token
        self._has_gaps = has_gaps

    def __str__(self) -> str:
        tagged_str = str(self.tagged) if isinstance(self.tagged, str) else ("\n" + 8 * " ").join(self.tagged)
        tokens_str = str(self.tokens).replace("\n", "\n" + 8 * " ")
        out_str = dedent(f"""\
          Text:
        {indent(self.text, 7 * " ")}
        Tagged:
        {indent(tagged_str, 7 * " ")}
        Tokens:
        {indent(tokens_str, 7 * " ")}
        """)
        if self.edits is not None:
            edits_tagged = cast(list[str], self.edits_tagged)
            edits_tokens = cast(ListOfListsOfLabeledToken, self.edits_tokens)
            aligned = cast(list[WordOutput], self.aligned)
            aligned_strings = self._get_aligned_strings(aligned)
            for idx, edit in enumerate(self.edits):
                edit_tagged_str = str(edits_tagged[idx])
                edit_tokens_str = str(edits_tokens[idx]).replace("\n", "\n" + 8 * " ").strip()
                aligned_str = aligned_strings[idx].replace("\n", "\n" + 8 * " ").strip()
                spans_str = str(self.spans[idx]).replace("\n", "\n" + 8 * " ")
                out_str += dedent(f"""\
        === Edit {idx} ===
        Edit Text:
        {indent(edit, 12 * " ")}
        Edit Tagged:
        {indent(edit_tagged_str, 12 * " ")}
        Edit Tokens:
        {indent(edit_tokens_str, 12 * " ")}
        Aligned:
        {indent(aligned_str, 12 * " ")}
        Spans:
        {indent(spans_str, 12 * " ")}
                """)
        else:
            spans_str = str(self.spans).replace("\n", "\n" + 8 * " ").strip()
            out_str += dedent(f"""\
            Spans:
            {indent(spans_str, 7 * " ")}
            """)
        return out_str.strip()

    ### Getters and Setters ###

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, t: str):
        raise RuntimeError("Cannot set the text after initialization")

    @property
    def edits(self) -> list[str] | None:
        return self._edits

    @edits.setter
    def edits(self, t: list[str]):
        raise RuntimeError("Cannot set the edited text after initialization")

    @property
    def spans(self) -> SpanList[Span] | ListOfListsOfSpans[EditSpan]:
        return self._spans

    @spans.setter
    def spans(self, s: SpanList[Span] | ListOfListsOfSpans[EditSpan]):
        raise RuntimeError("Cannot set the spans after initialization")

    @property
    def tagged(self) -> str | list[str]:
        return self._tagged

    @tagged.setter
    def tagged(self, t: str | list[str]):
        raise RuntimeError("Cannot set the tagged text after initialization")

    @property
    def tokens(self) -> LabeledTokenList | ListOfListsOfLabeledToken:
        return self._tokens

    @tokens.setter
    def tokens(self, t: LabeledTokenList | ListOfListsOfLabeledToken):
        raise RuntimeError("Cannot set the tokenized text after initialization")

    @property
    def tokens_offsets(self) -> list[tuple[int, int] | None]:
        return self._tokens_offsets

    @tokens_offsets.setter
    def tokens_offsets(self, t: list[tuple[int, int] | None]):
        raise RuntimeError("Cannot set the tokenized text offsets after initialization")

    @property
    def edits_tagged(self) -> list[str] | None:
        return self._edits_tagged

    @edits_tagged.setter
    def edits_tagged(self, t: list[str] | None):
        raise RuntimeError("Cannot set the tagged edited texts after initialization")

    @property
    def edits_tokens(self) -> ListOfListsOfLabeledToken | None:
        return self._edits_tokens

    @edits_tokens.setter
    def edits_tokens(self, t: ListOfListsOfLabeledToken | None):
        raise RuntimeError("Cannot set the tokenized edited text after initialization")

    @property
    def edits_tokens_offsets(self) -> list[list[tuple[int, int] | None]] | None:
        return self._edits_tokens_offsets

    @edits_tokens_offsets.setter
    def edits_tokens_offsets(self, t: list[list[tuple[int, int] | None]] | None):
        raise RuntimeError("Cannot set the tokenized edited text offsets after initialization")

    @property
    def has_bos_token(self) -> bool:
        return self._has_bos_token

    @has_bos_token.setter
    def has_bos_token(self, t: bool):
        raise RuntimeError("Cannot set the boolean flag `has_bos_token` after initialization")

    @property
    def has_eos_token(self) -> bool:
        return self._has_eos_token

    @has_eos_token.setter
    def has_eos_token(self, t: bool):
        raise RuntimeError("Cannot set the boolean flag `has_eos_token` after initialization")

    @property
    def has_gaps(self) -> bool | None:
        return self._has_gaps

    @has_gaps.setter
    def has_gaps(self, t: bool | None):
        raise RuntimeError("Cannot set the boolean flag `has_gaps` after initialization")

    ### Constructors ###

    @classmethod
    def from_edits(
        cls,
        text: str,
        edits: str | list[str],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
        with_gaps: bool = True,
        sub_label: str = "S",
        ins_label: str = "I",
        del_label: str = "D",
        gap_token: str = "▁",
    ) -> "WQEEntry":
        """Create a `WQEEntry` from a text and one or more edits.

        Args:
            text (str): The original text.
            edits (str | list[str] | None): One or more edited version of the text.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
            with_gaps (bool): Whether to add gaps to the tokens and offsets. Gaps are used to mark the positions of
                insertions and deletions in the original/edited texts, respectively. If false, those are merged to the
                next token to the right. Default: True.
            sub_label (str): The label for substitutions. Default: "S".
            ins_label (str): The label for insertions. Default: "I".
            del_label (str): The label for deletions. Default: "D".
            gap_token (str): The token to use for gaps. Default: "▁".

        Example:
            ```python
            from wqe.data.wqe_entry import WQEEntry

            entry = WQEEntry.from_edits(
                text="a simple example",
                edits=["this is a simple enough test, you know?", "an example"],
                tokenizer="facebook/nllb-200-3.3B",
                tokenizer_kwargs={
                    "tgt_lang": "ita_Latn",
                    "add_special_tokens": True,
                },
            )
            print(entry.aligned_str)
            >>> TEXT: ita_Latn ***** *** ▁a ▁simple ******* ***** * **** ***** ▁example </s>
                EDIT: ita_Latn ▁this ▁is ▁a ▁simple ▁enough ▁test , ▁you ▁know        ? </s>
                                I   I                  I     I I    I     I        S

                TEXT: ita_Latn  ▁a ▁simple ▁example </s>
                EDIT: ita_Latn ▁an ******* ▁example </s>
                                S       D
            ```
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        edits = [edits] if isinstance(edits, str) else edits
        aligned: list[WordOutput] | None = []
        tokens_str, offsets = tokenizer.tokenize_with_offsets(text)
        tokens_with_gaps, offsets_with_gaps = cls._add_gaps_to_tokens_and_offsets(
            tokens_str[0], offsets[0], tokenizer.has_bos_token, tokenizer.has_eos_token
        )
        all_edits_tokens_str, all_edits_offsets = tokenizer.tokenize_with_offsets(edits)
        all_edits_tokens_with_gaps, all_edits_offsets_with_gaps = cls._add_gaps_to_tokens_and_offsets(
            all_edits_tokens_str, all_edits_offsets, tokenizer.has_bos_token, tokenizer.has_eos_token
        )
        for edit_tokens_str in all_edits_tokens_str:
            aligned_out = process_words(
                texts=tokens_str, edits=[edit_tokens_str], is_text_pre_transformed=True, is_edit_pre_transformed=True
            )
            aligned.append(aligned_out)
        tokens, edits_tokens = cls.get_tokens_from_edits(
            text=text,
            edits=edits,
            tokens=tokens_with_gaps,
            offsets=offsets_with_gaps,
            edits_tokens=all_edits_tokens_with_gaps,
            edits_offsets=all_edits_offsets_with_gaps,
            aligned=aligned,
            tokenizer=tokenizer,
            has_tokens_gaps=True,
            sub_label=sub_label,
            ins_label=ins_label,
            del_label=del_label,
            gap_token=gap_token,
        )
        if not with_gaps:
            tokens = cls._merge_gap_annotations(tokens, has_bos_token=tokenizer.has_bos_token)
            edits_tokens = cls._merge_gap_annotations(edits_tokens, has_bos_token=tokenizer.has_bos_token)
            offsets = [offset + [None] for offset in offsets]
            all_edits_offsets = [edit_offsets + [None] for edit_offsets in all_edits_offsets]
        spans = cls.get_spans_from_edits(
            text=text,
            edits=edits,
            offsets=offsets[0],
            edits_offsets=all_edits_offsets,
            aligned=aligned,
            tokenizer=tokenizer,
            sub_label=sub_label,
            ins_label=ins_label,
            del_label=del_label,
        )
        f_orig_spans = cast(list[Span], [[s.orig for s in l if s.orig is not None] for l in spans])
        f_edit_spans = cast(list[Span], [[s.edit for s in l if s.edit is not None] for l in spans])
        tagged = cls.get_tagged_from_spans(texts=[text for _ in edits], spans=f_orig_spans)
        edits_tagged = cls.get_tagged_from_spans(texts=edits, spans=f_edit_spans)
        return cls(
            text=text,
            spans=spans,
            tagged=tagged,
            tokens=tokens,
            tokens_offsets=offsets_with_gaps if with_gaps else offsets[0],
            edits=edits,
            edits_tagged=edits_tagged,
            edits_tokens=edits_tokens,
            edits_tokens_offsets=all_edits_offsets_with_gaps if with_gaps else all_edits_offsets,
            aligned=aligned,
            has_bos_token=tokenizer.has_bos_token,
            has_eos_token=tokenizer.has_eos_token,
            has_gaps=with_gaps,
            constructor_key=cls.__constructor_key,
        )

    @classmethod
    def from_spans(
        cls,
        text: str,
        spans: list[Span] | list[dict[str, str | int | float | None]],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
    ) -> "WQEEntry":
        """Create a `WQEEntry` from a text and a list of spans.

        Args:
            text (str):
                The original text.
            spans (list[Span] | list[dict[str, str | int | float]]):
                A list of `Span` items containing information about specific spans in `text`.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        spans = Span.from_list(spans)
        tokens, offsets = cls.get_tokens_and_offsets_from_spans(texts=text, spans=spans, tokenizer=tokenizer)
        tagged = cls.get_tagged_from_spans(texts=text, spans=spans)[0]
        return cls(
            text=text,
            spans=spans,
            tagged=tagged,
            tokens=tokens[0],
            tokens_offsets=offsets[0],
            has_bos_token=tokenizer.has_bos_token,
            has_eos_token=tokenizer.has_eos_token,
            constructor_key=cls.__constructor_key,
        )

    @classmethod
    def from_tagged(
        cls,
        tagged: str,
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        keep_tags: list[str] = [],
        ignore_tags: list[str] = [],
        tokenizer_kwargs: dict = {},
    ) -> "WQEEntry":
        """Create a `WQEEntry` from a tagged text.

        Args:
            tagged (str): Tagged version of `text` containing information from `spans`.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            keep_tags (list[str]):
                Tag(s) used to mark selected spans, e.g. `h` for tags like `<h>...</h>`. If not provided, all
                tags are kept (Default: []).
            ignore_tags (list[str]):
                Tag(s) that are present in the text but should be ignored while parsing. If not provided, all tags
                are kept (Default: []).
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        text, spans = cls.get_text_and_spans_from_tagged(tagged=tagged, keep_tags=keep_tags, ignore_tags=ignore_tags)
        tokens, offsets = cls.get_tokens_and_offsets_from_spans(texts=text, spans=spans, tokenizer=tokenizer)
        return cls(
            text=text,
            spans=spans,
            tagged=tagged,
            tokens=tokens[0],
            tokens_offsets=offsets[0],
            has_bos_token=tokenizer.has_bos_token,
            has_eos_token=tokenizer.has_eos_token,
            constructor_key=cls.__constructor_key,
        )

    @classmethod
    def from_tokens(
        cls,
        labeled_tokens: LabeledTokenInput | None = None,
        keep_labels: list[str] = [],
        ignore_labels: list[str] = [],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
        tokens: list[str] | None = None,
        labels: Sequence[str | int | float | None] | None = None,
    ) -> "WQEEntry":
        """Create a `WQEEntry` from a list of tokens.

        Args:
            labeled_tokens (list[tuple[str, str]] | list[tuple[str, int]] | list[tuple[str, float]] | list[LabeledToken]):
                Tokenized variant of `text` with text or numeric labels for each token.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            keep_labels (list[str]):
                Label(s) used to mark selected tokens. If not provided, all labels are kept (Default: []).
            ignore_labels (list[str]):
                Label(s) that are present on tokens but should be ignored while parsing. If not provided, all labels
                are kept (Default: []).
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
            tokens (list[str] | None):
                A list of tokens. Can be provided together with `labels` as an alternative to `labeled_tokens`.
            labels (list[str | int | float | None] | None):
                A list of labels for the tokens. Can be provided together with `tokens` as an alternative to
                `labeled_tokens`.

        Example:
            ```python
            from wqe.data.wqe_entry import WQEEntry

            entry = WQEEntry.from_tokens(
                labeled_tokens=[
                    ("Apple", "ORG"), ("Inc.", "ORG"), ("is", "O"), ("looking", "O"),
                    ("at", "O"), ("buying", "O"), ("U.K.", "LOC"), ("startup", "O"),
                    ("for", "O"), ("$1", "MONEY"), ("billion", "MONEY")
                ],
                ignore_labels=["O"],
            )
            print(entry.tokens)
            >>> Apple Inc. is looking at buying U.K. startup for    $1 billion
                  ORG  ORG                       LOC             MONEY   MONEY
            ```
        """
        if labeled_tokens and (tokens or labels):
            raise RuntimeError(
                "Cannot provide both `labeled_tokens` and `tokens`/`labels`. "
                "Use `labeled_tokens` to specify the tokenized text with labels."
            )
        if tokens and not labels or labels and not tokens:
            raise RuntimeError(
                "If `tokens` is provided, `labels` must also be provided. "
                "Use `labeled_tokens` to specify the tokenized text with labels instad."
            )
        if tokens and labels:
            if len(tokens) != len(labels):
                raise RuntimeError("The length of `tokens` and `labels` must be the same. ")
            labeled_tokens = [(tok, lab) for tok, lab in zip(tokens, labels, strict=True)]  # type: ignore
        labeled_tokens = LabeledToken.from_list(labeled_tokens, keep_labels=keep_labels, ignore_labels=ignore_labels)  # type: ignore
        labeled_tokens = cast(LabeledTokenList, labeled_tokens)
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        text = tokenizer.detokenize([tok.t for tok in labeled_tokens])[0]
        _, offsets = tokenizer.tokenize_with_offsets(text)
        spans = cls.get_spans_from_tokens(text, labeled_tokens, offsets[0], tokenizer, keep_labels, ignore_labels)
        tagged = cls.get_tagged_from_spans(text, spans=spans)[0]
        return cls(
            text=text,
            spans=spans,
            tagged=tagged,
            tokens=labeled_tokens,
            tokens_offsets=offsets[0],
            has_bos_token=tokenizer.has_bos_token,
            has_eos_token=tokenizer.has_eos_token,
            constructor_key=cls.__constructor_key,
        )

    ### Helper Functions ###

    @overload
    @staticmethod
    def _add_gaps_to_tokens_and_offsets(
        tokens: list[str],
        offsets: list[tuple[int, int] | None],
        has_bos_token: bool,
        has_eos_token: bool,
        gap_token: str = "▁",
    ) -> tuple[list[str], list[tuple[int, int] | None]]: ...

    @overload
    @staticmethod
    def _add_gaps_to_tokens_and_offsets(
        tokens: list[list[str]],
        offsets: list[list[tuple[int, int] | None]],
        has_bos_token: bool,
        has_eos_token: bool,
        gap_token: str = "▁",
    ) -> tuple[list[list[str]], list[list[tuple[int, int] | None]]]: ...

    @staticmethod
    def _add_gaps_to_tokens_and_offsets(
        tokens: list[str] | list[list[str]],
        offsets: list[tuple[int, int] | None] | list[list[tuple[int, int] | None]],
        has_bos_token: bool,
        has_eos_token: bool,
        gap_token: str = "▁",
    ) -> tuple[list[str], list[tuple[int, int] | None]] | tuple[list[list[str]], list[list[tuple[int, int] | None]]]:
        """Adds gaps to a sequence of tokens and their offsets.

        This is useful for adding annotations of insertions and deletions to the text and edits, respectively.
        The resulting sequence will have 2N + 1 `tokens`, with `gap_token` at even indices like:
        `GAP Hello GAP World GAP ! GAP`. The `offsets` will be `None` for the gaps.

        Args:
            tokens (list[str]): The tokens to add gaps to.
            offsets (list[tuple[int, int]  |  None]): The offsets of the tokens.
            has_bos_token (bool): Whether the token sequence has a beginning-of-sequence token.
            has_eos_token (bool): Whether the token sequence has an end-of-sequence token.
            gap_token (str): The token to use for gaps.

        Returns:
            The tokens and offsets with gaps added.
        """
        single_list = False
        if isinstance(tokens, list) and not isinstance(tokens[0], list):
            tokens = cast(list[list[str]], [tokens])
            single_list = True
        if isinstance(offsets, list) and not isinstance(offsets[0], list):
            offsets = cast(list[list[tuple[int, int] | None]], [offsets])
            single_list = True
        tokens_with_gaps = []
        offsets_with_gaps = []
        for curr_tokens, curr_offsets in zip(tokens, offsets, strict=True):
            curr_offsets = cast(list[tuple[int, int] | None], curr_offsets)
            curr_tokens_with_gaps = []
            curr_offsets_with_gaps = []
            for idx, (tok, off) in enumerate(zip(curr_tokens, curr_offsets, strict=True)):
                if idx == 0:
                    if not has_bos_token:
                        curr_tokens_with_gaps.append(gap_token)
                        curr_offsets_with_gaps.append(None)
                    else:
                        curr_tokens_with_gaps.append(tok)
                        curr_offsets_with_gaps.append(off)
                        continue
                curr_tokens_with_gaps.append(tok)
                curr_offsets_with_gaps.append(off)
                if (idx < len(curr_tokens) - 2 and has_eos_token) or not has_eos_token:
                    curr_tokens_with_gaps.append(gap_token)
                    curr_offsets_with_gaps.append(None)
            tokens_with_gaps.append(curr_tokens_with_gaps)
            offsets_with_gaps.append(curr_offsets_with_gaps)

        if len(tokens_with_gaps) == 1 and single_list:
            tokens_with_gaps = tokens_with_gaps[0]
            offsets_with_gaps = offsets_with_gaps[0]
        return tokens_with_gaps, offsets_with_gaps

    @staticmethod
    def _span_from_offsets_indices(
        text: str,
        token_start_idx: int,
        token_end_idx: int,
        offsets: list[tuple[int, int] | None],
        label: str | int | float | None,
    ) -> Span:
        """Creates a `Span` by extracting the text between the start and end indices of the tokens in the offsets.

        Args:
            text (str): The text to extract the span from.
            token_start_idx (int): The index of the token where the span starts.
            token_end_idx (int): The index of the token where the span ends.
            offsets (list[tuple[int, int]  |  None]): A list of offsets for each token. Offsets are tuples of (start, end) indices marking
                the position of the token in the text. If a token does not appear in the original text, its offset is `None`.
            label (str | int | float | None): The label for the span.

        Returns:
            A `Span` object containing the start and end indice in `text`, the corresponding string and a label.
        """
        start_span = offsets[token_start_idx]
        text_start_idx = start_span[0] if start_span is not None else -1
        end_span = offsets[token_end_idx - 1]
        text_end_idx = end_span[1] if end_span is not None else -1
        span_text = text[text_start_idx:text_end_idx] if start_span is not None and end_span is not None else None
        return Span(start=text_start_idx, end=text_end_idx, label=label, text=span_text)

    @staticmethod
    def _merge_gap_annotations(
        tokens: ListOfListsOfLabeledToken,
        has_bos_token: bool,
    ) -> ListOfListsOfLabeledToken:
        """Merges gap annotations in a list of tokens.

        Args:
            tokens (ListOfListsOfLabeledToken): The list of tokens to merge.
            has_bos_token (bool): Whether the token sequence has a beginning-of-sequence token.
        Returns:
            A list of tokens with going from 2N + 1 tokens (assuming no bos/eos) to N + 1 tokens
            (only the last gap is kept to handle end-of-sequence insertions).
        """
        merged_tokens = ListOfListsOfLabeledToken()
        for token_list in tokens:
            merged_token_list = LabeledTokenList()
            gap_label = None
            for idx, token in enumerate(token_list):
                if idx % 2 == 0:  # Even indices are gaps
                    # Final gap is kept regardless of it being an EOS token or not
                    if (has_bos_token and idx == 0) or idx == len(token_list) - 1:
                        merged_token_list.append(token)
                    else:
                        gap_label = token.label
                else:
                    if gap_label is not None and token.l is not None:
                        label = token.l + gap_label  # type: ignore
                    elif token.l is None:
                        label = gap_label
                    else:
                        label = token.l
                    merged_token_list.append(LabeledToken(token=token.t, label=label))
                    gap_label = None
            merged_tokens.append(merged_token_list)
        return merged_tokens

    @staticmethod
    def _remove_gap_offsets(
        all_offsets: list[list[tuple[int, int] | None]],
        has_bos_token: bool,
    ) -> list[list[tuple[int, int] | None]]:
        """Removes gap offsets from a list of offsets.
        Args:
            all_offsets (list[list[tuple[int, int] | None]]): The list of offsets to filter.
            has_bos_token (bool): Whether the token sequence has a beginning-of-sequence token.
        Returns:
            A list of offsets with gaps removed. The final gap offset is kept regardless of it being an EOS token
            to handle end-of-sequence insertions.
        """
        return [
            [
                off
                for idx, off in enumerate(offsets)
                if off is not None or idx == len(offsets) - 1 or (idx == 0 and has_bos_token)
            ]
            for offsets in all_offsets
        ]

    ### Formatting Methods ###

    @staticmethod
    def get_spans_from_edits(
        text: str,
        edits: str | list[str],
        offsets: list[tuple[int, int] | None] | None = None,
        edits_offsets: list[list[tuple[int, int] | None]] | None = None,
        aligned: WordOutput | list[WordOutput] | None = None,
        tokenizer: Tokenizer | None = None,
        sub_label: str = "S",
        ins_label: str = "I",
        del_label: str = "D",
    ) -> ListOfListsOfSpans[EditSpan]:
        """Convert edits to spans over a text and its edits.

        Args:
            text (str): The text.
            edits (str | list[str]): The edited text(s).
            aligned (WordOutput | list[WordOutput] | None): The aligned
                `jiwer` output(s) between text and each one of its edits.
                If `edits` is a single string, this should be a single `WordOutput`. If `edits` is a list of
                strings, this should be a list of `WordOutput` objects. If not provided, the function will
                compute the character-level alignment automatically.
            sub_label (str): The label for substitutions. Default: "S".
            ins_label (str): The label for insertions. Default: "I".
            del_label (str): The label for deletions. Default: "D".

        Returns:
            One or more lists of `EditSpan` objects with edit labels.
        """
        if isinstance(edits, str):
            edits = [edits]
        if tokenizer is None:
            logger.info("Tokenizer was not provided. Defaulting to whitespace tokenization.")
            tokenizer = WhitespaceTokenizer()
        if aligned is None:
            aligned = [
                process_words(text, edit, texts_transform=tokenizer.transform, edits_transform=tokenizer.transform)
                for edit in edits
            ]
        elif isinstance(aligned, WordOutput):
            aligned = cast(list[WordOutput], [aligned])
        else:
            aligned = cast(list[WordOutput], aligned)
        if offsets is None:
            _, all_offsets = tokenizer.tokenize_with_offsets(text)
            offsets = all_offsets[0]
        if edits_offsets is None:
            _, edits_offsets = tokenizer.tokenize_with_offsets(edits)
        all_spans: ListOfListsOfSpans[EditSpan] = ListOfListsOfSpans()
        for edit, edit_offsets, al in zip(edits, edits_offsets, aligned, strict=True):
            edit_spans: SpanList[EditSpan] = SpanList()
            for aligned_span in al.alignments[0]:
                if aligned_span.type != "equal":
                    orig_span = None
                    if aligned_span.type in ("delete", "substitute"):
                        orig_span = WQEEntry._span_from_offsets_indices(
                            text=text,
                            token_start_idx=aligned_span.ref_start_idx,
                            token_end_idx=aligned_span.ref_end_idx,
                            offsets=offsets,
                            label=sub_label if aligned_span.type == "substitute" else del_label,
                        )
                    edit_span = None
                    if aligned_span.type in ("insert", "substitute"):
                        edit_span = WQEEntry._span_from_offsets_indices(
                            text=edit,
                            token_start_idx=aligned_span.hyp_start_idx,
                            token_end_idx=aligned_span.hyp_end_idx,
                            offsets=edit_offsets,
                            label=sub_label if aligned_span.type == "substitute" else ins_label,
                        )
                    if orig_span or edit_span:
                        edit_spans.append(EditSpan(orig=orig_span, edit=edit_span))
            all_spans.append(edit_spans)
        return all_spans

    @staticmethod
    def get_tokens_from_edits(
        text: str,
        edits: str | list[str],
        tokens: list[str] | None = None,
        offsets: list[tuple[int, int] | None] | None = None,
        edits_tokens: list[list[str]] | None = None,
        edits_offsets: list[list[tuple[int, int] | None]] | None = None,
        aligned: WordOutput | list[WordOutput] | None = None,
        tokenizer: Tokenizer | None = None,
        has_tokens_gaps: bool = False,
        sub_label: str = "S",
        ins_label: str = "I",
        del_label: str = "D",
        gap_token: str = "▁",
    ) -> tuple[ListOfListsOfLabeledToken, ListOfListsOfLabeledToken]:
        """Convert edits to tokens.

        Args:
            text (str): The text.
            edits (str | list[str]): The edited text(s).
            aligned (WordOutput | list[WordOutput] | None): The aligned output
                If `edits` is a single string, this should be a single `WordOutput`. If `edits` is a list of
                strings, this should be a list of `WordOutput` objects. If not provided, the function will
                obtain it automatically using the tokenizer for spltting. Default: None.
            tokenizer (Tokenizer | None): A `Tokenizer` used for text splitting. If not provided, whitespace
                tokenization is used. Default: None.

        Returns:
            One or more lists of `LabeledToken` objects with edit tags and their offsets.
        """
        if isinstance(edits, str):
            edits = [edits]
        if tokenizer is None:
            logger.info("Tokenizer was not provided. Defaulting to whitespace tokenization.")
            tokenizer = WhitespaceTokenizer()
        if aligned is None:
            aligned = [
                process_words(text, edit, texts_transform=tokenizer.transform, edits_transform=tokenizer.transform)
                for edit in edits
            ]
        elif isinstance(aligned, WordOutput):
            aligned = cast(list[WordOutput], [aligned])
        else:
            aligned = cast(list[WordOutput], aligned)
        if tokens is None or offsets is None:
            all_tokens_str, all_offsets = tokenizer.tokenize_with_offsets(text)
            tokens = all_tokens_str[0]
            offsets = all_offsets[0]
        if edits_tokens is None or edits_offsets is None:
            edits_tokens, edits_offsets = tokenizer.tokenize_with_offsets(edits)
        if not has_tokens_gaps:
            tokens, offsets = WQEEntry._add_gaps_to_tokens_and_offsets(
                tokens, offsets, tokenizer.has_bos_token, tokenizer.has_eos_token, gap_token=gap_token
            )
            edits_tokens, edits_offsets = WQEEntry._add_gaps_to_tokens_and_offsets(
                edits_tokens, edits_offsets, tokenizer.has_bos_token, tokenizer.has_eos_token, gap_token=gap_token
            )
        all_tokens: ListOfListsOfLabeledToken = ListOfListsOfLabeledToken()
        all_edits_tokens: ListOfListsOfLabeledToken = ListOfListsOfLabeledToken()
        for output, curr_edit_tokens in zip(aligned, edits_tokens, strict=True):
            token_labels: list[str | None] = [None] * len(tokens)
            edit_labels: list[str | None] = [None] * len(curr_edit_tokens)
            for alignment in output.alignments[0]:
                text_start_idx = alignment.ref_start_idx
                text_end_idx = alignment.ref_end_idx
                edit_start_idx = alignment.hyp_start_idx
                edit_end_idx = alignment.hyp_end_idx
                if tokenizer.has_bos_token:
                    text_start_idx -= 1
                    text_end_idx -= 1
                    edit_start_idx -= 1
                    edit_end_idx -= 1
                if alignment.type == "insert":
                    token_labels[text_start_idx * 2] = ins_label
                elif alignment.type in ("delete", "substitute"):
                    label = sub_label if alignment.type == "substitute" else del_label
                    for idx in range(text_start_idx, text_end_idx):
                        token_labels[idx * 2 + 1] = label
                if alignment.type == "delete":
                    edit_labels[edit_start_idx * 2] = del_label
                elif alignment.type in ("insert", "substitute"):
                    label = sub_label if alignment.type == "substitute" else ins_label
                    for idx in range(edit_start_idx, edit_end_idx):
                        edit_labels[idx * 2 + 1] = label
            tokens_with_labels = LabeledTokenList(
                [LabeledToken(tok, label) for tok, label in zip(tokens, token_labels, strict=True)]
            )
            edits_tokens_with_labels = LabeledTokenList(
                [LabeledToken(tok, label) for tok, label in zip(curr_edit_tokens, edit_labels, strict=True)]
            )
            all_tokens.append(tokens_with_labels)
            all_edits_tokens.append(edits_tokens_with_labels)
        return all_tokens, all_edits_tokens

    @staticmethod
    def get_tagged_from_spans(
        texts: str | list[str],
        spans: list[Span] | list[list[Span]],
    ) -> list[str]:
        """Tags one or more texts using lists of spans.

        Args:
            texts (str | list[str]): The text to which tags should be added.
            spans (list[Span] | list[list[Span]]): The spans to convert to tags.

        Returns:
            A string or a list of strings representing tagged texts.
        """
        if isinstance(texts, str):
            texts = [texts]
        if not spans:
            return texts
        if not isinstance(spans[0], list):
            spans = [spans]  # type: ignore
        spans = cast(list[list[Span]], spans)
        tagged_edits = []
        for text, span_list in zip(texts, spans, strict=True):
            tagged = text
            sorted_span_list = sorted(span_list, key=lambda s: s.start)
            offset = 0
            for s in sorted_span_list:
                if s.label:
                    start = s.start + offset
                    end = s.end + offset
                    label = s.label
                    tagged = f"{tagged[:start]}<{label}>{tagged[start:end]}</{label}>{tagged[end:]}"

                    # Update the offset for the next span
                    offset += len(str(label)) * 2 + 5  # <{label}>...</{label}>
            tagged_edits.append(tagged)
        return tagged_edits

    @staticmethod
    def get_tokens_and_offsets_from_spans(
        texts: str | list[str],
        spans: list[Span] | list[list[Span]],
        tokenizer: Tokenizer | None = None,
    ) -> tuple[ListOfListsOfLabeledToken, list[list[tuple[int, int] | None]]]:
        """Convert spans to tokens.

        Args:
            texts (str | list[str]): The text(s) to which tags should be added.
            spans (list[Span] | list[list[Span]]): The spans to convert to tokens.
            tokenizer (Tokenizer | None): A `Tokenizer` used for text splitting. If not provided, whitespace
                tokenization is used.

        Returns:
            One or more lists of `LabeledToken` representing the tagged texts.
        """
        if isinstance(texts, str):
            texts = [texts]
        if tokenizer is None:
            logger.info("Tokenizer was not provided. Defaulting to whitespace tokenization.")
            tokenizer = WhitespaceTokenizer()
        all_str_tokens, all_offsets = tokenizer.tokenize_with_offsets(texts)
        if not spans:
            out = ListOfListsOfLabeledToken(
                LabeledTokenList([LabeledToken(tok, None) for tok in str_tokens]) for str_tokens in all_str_tokens
            )
            return out, all_offsets
        if not isinstance(spans[0], list):
            spans = [spans]  # type: ignore
        spans = cast(list[list[Span]], spans)
        all_labeled_tokens = ListOfListsOfLabeledToken()
        for tokens, offsets, l_spans in zip(all_str_tokens, all_offsets, spans, strict=True):
            labeled_tokens = LabeledTokenList()
            sorted_l_spans = sorted(l_spans, key=lambda s: s.start)

            # Pointer for the current position in sorted_l_spans
            span_idx = 0

            for i in range(len(tokens)):
                token = tokens[i]
                offset = offsets[i]

                if offset is None:
                    labeled_tokens.append(LabeledToken(token, None))
                    continue

                token_start, token_end = offset

                token_label = None

                # Skip spans that end before the token starts
                while span_idx < len(sorted_l_spans) and sorted_l_spans[span_idx].end <= token_start:
                    span_idx += 1

                # Iterate through spans starting from the current span_idx, as long as
                # the span starts before the current token ends. If a span starts
                # at or after the token ends, it (and all subsequent spans) cannot overlap.
                current_check_idx = span_idx
                while current_check_idx < len(sorted_l_spans) and sorted_l_spans[span_idx].start < token_end:
                    span = sorted_l_spans[current_check_idx]

                    # Check for actual overlap using the standard condition:
                    # Does the interval [token_start, token_end) intersect with [span_start, span_end)?
                    # Overlap = max(start1, start2) < min(end1, end2)
                    if max(token_start, span.start) < min(token_end, span.end):
                        if token_label is None:
                            token_label = span.label
                        else:
                            token_label += span.label  # type: ignore
                    current_check_idx += 1  # Move to the next potentially overlapping span
                labeled_tokens.append(LabeledToken(token, token_label))
            all_labeled_tokens.append(labeled_tokens)
        return all_labeled_tokens, all_offsets

    @staticmethod
    def get_text_and_spans_from_tagged(
        tagged: str,
        keep_tags: list[str] = [],
        ignore_tags: list[str] = [],
    ) -> tuple[str, SpanList[Span]]:
        """Extract spans and clean text from a tagged string.

        Args:
            tagged (str): The tagged string to extract spans from.
            keep_tags (list[str]):
                Tag(s) used to mark selected spans, e.g. `<h>...</h>`, `<error>...</error>`. If not provided,
                all tags are kept (Default: []).
            ignore_tags (list[str]):
                Tag(s) that are present in the text but should be ignored while parsing. If not provided,
                all tags are kept (Default: []).

        Returns:
            Tuple containing the cleaned text and a list of `Span` objects.
        """
        any_tag_regex = re.compile(r"<\/?(?:\w+)>")
        if not keep_tags:
            tag_regex = any_tag_regex
        else:
            tag_match_string = "|".join(list(set(keep_tags) | set(ignore_tags)))
            tag_regex = re.compile(rf"<\/?(?:{tag_match_string})>")

        text_without_tags: str = ""
        span_dicts: SpanList[Span] = SpanList()
        current_pos = 0
        open_tags = []
        open_positions = []

        for match in tag_regex.finditer(tagged):
            match_text = match.group(0)
            start, end = match.span()

            # Add text before the tag
            text_without_tags += tagged[current_pos:start]
            current_pos = end

            # Check if opening or closing tag
            if match_text.startswith("</"):
                tag_name = match_text[2:-1]
                if not open_tags or open_tags[-1] != tag_name:
                    raise RuntimeError(f"Closing tag {match_text} without matching opening tag")

                # Create span for the highlighted text
                open_pos = open_positions.pop()
                open_tag = open_tags.pop()
                if tag_name not in ignore_tags:
                    tagged_span = Span(
                        start=open_pos,
                        end=len(text_without_tags),
                        label=open_tag,
                    )
                    span_dicts.append(tagged_span)
            else:
                # Opening tag
                tag_name = match_text[1:-1]
                if keep_tags and (tag_name not in keep_tags and tag_name not in ignore_tags):
                    raise RuntimeError(
                        f"Unexpected tag type: {tag_name}. "
                        "Specify tag types that should be preserved in the `keep_tags` argument, "
                        "and those that should be ignored in the `ignore_tag_types` argument."
                    )
                open_tags.append(tag_name)
                open_positions.append(len(text_without_tags))

        # Add remaining text
        text_without_tags += tagged[current_pos:]
        if open_tags:
            raise RuntimeError(f"Unclosed tags: {', '.join(open_tags)}")

        # If the text contains a tag that was neither kept nor ignored, raise a warning
        unexpected_tags = any_tag_regex.search(text_without_tags)
        if unexpected_tags:
            warn(
                "The text contains tag types that were not specified in keep_tags or ignore_tags: "
                f"{unexpected_tags.group(0)}. These tags are now preserved in the output. If these should ignored "
                "instead, add them to the `ignore_tags` argument.",
                stacklevel=2,
            )
        for span in span_dicts:
            span.text = text_without_tags[span.start : span.end]
        return text_without_tags, span_dicts

    @staticmethod
    def get_spans_from_tokens(
        text: str,
        tokens: LabeledTokenInput,
        offsets: list[tuple[int, int] | None] | None = None,
        tokenizer: Tokenizer | None = None,
        keep_labels: list[str] = [],
        ignore_labels: list[str] = [],
    ) -> SpanList[Span]:
        """Extract spans and clean text from a list of labeled tokens.

        Args:
            tokens (list[tuple[str, str]] | list[tuple[str, int]] | list[tuple[str, float]] | list[LabeledToken]):
                The tokenized text to extract spans from.
            tokenizer (Tokenizer | None): The tokenizer to use for
                tokenization. If not provided, whitespace tokenization is used.
            keep_labels (list[str]):
                Token labels that should be ported over to spans. If not provided, all tags are kept (Default: []).
            ignore_labels (list[str]):
                Token labels that should be ignored while parsing. If not provided, all tags are kept (Default: []).

        Returns:
            A list of `Span` objects corresponding to the labeled tokens.
        """
        tokens = LabeledToken.from_list(tokens)
        if offsets is None:
            if tokenizer is None:
                logger.info("Tokenizer was not provided. Defaulting to whitespace tokenization.")
                tokenizer = WhitespaceTokenizer()
            _, all_offsets = tokenizer.tokenize_with_offsets(text)
            offsets = all_offsets[0]
        curr_span_label: str | int | float | None = None
        curr_span_start: int | None = None
        curr_span_end: int | None = None
        spans: SpanList[Span] = SpanList()

        # To be considered for a span, a token must have a valid label (not ignored) and a valid character span
        # (not a special token).
        for tok, offset in zip(tokens, offsets, strict=True):
            is_ignored = tok.l in ignore_labels
            is_kept = not keep_labels or tok.l in keep_labels
            has_valid_label = is_kept and not is_ignored

            if has_valid_label and offset is not None:
                t_start, t_end = offset
                if tok.l == curr_span_label:
                    curr_span_end = t_end
                else:
                    if curr_span_label is not None and curr_span_start is not None and curr_span_end is not None:
                        spans.append(Span(start=curr_span_start, end=curr_span_end, label=curr_span_label))
                    curr_span_label = tok.l
                    curr_span_start = t_start
                    curr_span_end = t_end
            else:
                if curr_span_label is not None and curr_span_start is not None and curr_span_end is not None:
                    spans.append(Span(start=curr_span_start, end=curr_span_end, label=curr_span_label))
                curr_span_label = None
                curr_span_start = None
                curr_span_end = None
        if curr_span_label is not None and curr_span_start is not None and curr_span_end is not None:
            spans.append(Span(start=curr_span_start, end=curr_span_end, label=curr_span_label))
        for span in spans:
            span.text = text[span.start : span.end]
        return spans

    ### Analysis Methods ###

    def merge_gap_annotations(self):
        """Merge gap annotations in the `tokens`, `tokens_offsets`, `edits_tokens` and `edits_offsets` attributes.

        This method is equivalent to calling `WQEEntry.from_edits` with `with_gaps=False`. Gap annotations are merged
        to the next non-gap token to the right, and the gap label is added to the label of the non-gap token. The last
        gap is kept to account for insertions at the end of the text.

        E.g. `GAP Hello GAP World GAP ! GAP` becomes `Hello World ! GAP`.
             `  I     S   I               I`         `   IS     I     I`
        """
        if self.has_gaps is None:
            raise RuntimeError("Gaps are not available for entries that were not initialized from edits.")
        elif self.has_gaps is False:
            raise RuntimeError("Gaps for the current entry were already merged.")
        if isinstance(self.tokens, ListOfListsOfLabeledToken):
            self._tokens = self._merge_gap_annotations(self.tokens, self.has_bos_token)
            self._tokens_offsets = self._remove_gap_offsets([self.tokens_offsets], self.has_bos_token)[0]
        if self.edits_tokens is not None and self.edits_tokens_offsets is not None:
            self._edits_tokens = self._merge_gap_annotations(self.edits_tokens, self.has_bos_token)
            self._edits_tokens_offsets = self._remove_gap_offsets(self.edits_tokens_offsets, self.has_bos_token)
        self._has_gaps = False

    def token_labels_summary(
        self,
        aggregation: LabelAggregation = label_sumlen_aggregation,
    ) -> LabeledTokenList:
        """If multiple `tokens` sequences are present, e.g. from multiple edits, get a summary of labels present on the
        `tokens` with a customizable aggregation.

        Args:
            aggregation (Callable[[Sequence[Any], ...], Any]): The aggregation method to use for the summary.
                Default: Total length of non-empty labels.

        Returns:
            A list of `LabeledToken` objects with the aggregated labels.
        """
        if isinstance(self.tokens, LabeledTokenList):
            raise RuntimeError("Cannot summarize labels from a single list of tokens.")
        summarized = LabeledTokenList()
        tok_labels_variants: zip[tuple[LabeledToken, ...]] = zip(*self.tokens, strict=True)
        for tok_labels in tok_labels_variants:
            token = tok_labels[0].t
            aggregate_label = aggregation([tok.l for tok in tok_labels])
            summarized.append(LabeledToken(token, aggregate_label))
        return summarized

    def span_labels_summary(
        self,
        aggregation: LabelAggregation = label_sumlen_aggregation,
    ) -> SpanList[Span]:
        """If multiple `spans` sequences are present, e.g. from multiple edits, get a summary of labels present on the
        `spans` with a customizable aggregation.

        Args:
            aggregation (Callable[[Sequence[Any], ...], Any]): The aggregation method to use for the summary.
                Default: Total length of non-empty labels.

        Returns:
            A list of `Span` objects with the aggregated labels.
        """
        if isinstance(self.spans, SpanList):
            raise RuntimeError("Cannot summarize labels from a single list of spans.")
        summarized_tokens = self.token_labels_summary(aggregation=aggregation)
        return self.get_spans_from_tokens(self.text, summarized_tokens, self.tokens_offsets)
