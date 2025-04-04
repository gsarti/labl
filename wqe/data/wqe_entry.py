import re
from collections.abc import Sequence
from logging import getLogger
from textwrap import dedent, indent
from typing import Literal, cast, overload
from warnings import warn

from jiwer import CharacterOutput, WordOutput, process_characters, process_words
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from wqe.data.mixin import AlignedSequencesMixin
from wqe.data.span import EditSpan, QESpanInput, QESpanWithEditInput, Span
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
        edits (list[str] | None):
            One or more edited version of the text. If only the text was provided, this will be set to None.
        spans (list[Span] | list[list[EditSpan]]): If only `text` is specified, `spans` is a list of
            `Span` items containing information about specific spans in `text`. If one or more `edits` are provided,
            `spans` is a list where the i-th element contains a list `EditSpan` with information about `text` and
            the corresponding i-th edit in `edits`.
        tagged (str | list[str]): Tagged version of `text` containing information from `spans`. If multiple edits
            are provided, `tagged` is a list where the i-th element contains the tagged version of `text` with
            information from the i-th edit in `edits`.
        edits_tagged (list[str]): Tagged version of `edits` containing information from `spans`, if present.
        tokens (LabeledTokenList | ListOfListsOfLabeledToken): Tokenized variant of `text` with text or numeric
            labels for each token. If multiple edits are provided, `tokens` is a list where the i-th element
            contains a tokenized version of `text` with text or numeric labels for each token.
        edits_tokens (ListOfListsOfLabeledToken): List where the i-th element contains a tokenized version of the i-th
            edit in `edits` with text or numeric labels for each token.
        aligned (list[WordOutput] | None): If one or more edits are provided, this is a list of aligned WordOutputs
            (one per edit) tokenized using the provided tokenizer. The alignment is done using the `jiwer` library.
        aligned_char (list[CharacterOutput] | None): If one or more edits are provided, this is a list of aligned
            CharacterOutputs (one per edit). The alignment is done using the `jiwer` library.
    """

    __constructor_key = object()

    def __init__(
        self,
        text: str,
        spans: list[Span] | list[list[EditSpan]],
        tagged: str | list[str],
        tokens: LabeledTokenList | ListOfListsOfLabeledToken,
        edits: list[str] | None = None,
        edits_tagged: list[str] | None = None,
        edits_tokens: ListOfListsOfLabeledToken | None = None,
        aligned: list[WordOutput] | None = None,
        aligned_char: list[CharacterOutput] | None = None,
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
        self._edits = edits
        self._edits_tagged = edits_tagged
        self._edits_tokens = edits_tokens
        self._aligned = aligned
        self._aligned_char = aligned_char

    def __str__(self) -> str:
        if self.aligned is not None:
            return self._get_aligned_string(self.aligned, add_stats=True)
        tokens_str = str(self.tokens).replace("\n", "\n" + 8 * " ")
        return dedent(f"""\
          Text:
        {indent(self.text, 7 * " ")}
        Tagged:
        {indent(str(self.tagged), 7 * " ")}
        Tokens:
        {indent(tokens_str, 7 * " ")}
        """)

    def __repr__(self) -> str:
        return self.__str__()

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
    def edits(self, t: str):
        raise RuntimeError("Cannot set the edited text after initialization")

    @property
    def spans(self) -> list[Span] | list[list[EditSpan]] | None:
        return self._spans

    @spans.setter
    def spans(self, s: list[Span] | list[list[EditSpan]] | None):
        raise RuntimeError("Cannot set the spans after initialization")

    @property
    def tagged(self) -> str | list[str] | None:
        return self._tagged

    @tagged.setter
    def tagged(self, t: str | None):
        raise RuntimeError("Cannot set the tagged text after initialization")

    @property
    def edits_tagged(self) -> list[str] | None:
        return self._edits_tagged

    @edits_tagged.setter
    def edits_tagged(self, t: list[str] | None):
        raise RuntimeError("Cannot set the tagged edited texts after initialization")

    @property
    def tokens(self) -> LabeledTokenList | ListOfListsOfLabeledToken | None:
        return self._tokens

    @tokens.setter
    def tokens(self, t: LabeledTokenList | ListOfListsOfLabeledToken | None):
        raise RuntimeError("Cannot set the tokenized text after initialization")

    @property
    def edits_tokens(self) -> ListOfListsOfLabeledToken | None:
        return self._edits_tokens

    @edits_tokens.setter
    def edits_tokens(self, t: ListOfListsOfLabeledToken | None):
        raise RuntimeError("Cannot set the tokenized edited text after initialization")

    ### Constructors ###

    @classmethod
    def from_edits(
        cls,
        text: str,
        edits: str | list[str],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
    ) -> "WQEEntry":
        """Create a `WQEEntry` from a text and one or more edits.

        Args:
            text (str):
                The original text.
            edits (str | list[str] | None):
                One or more edited version of the text.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A
                [jiwer transform](https://jitsi.github.io/jiwer/reference/transforms/#transforms)
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        edits = [edits] if isinstance(edits, str) else edits
        aligned: list[WordOutput] | None = []
        aligned_char: list[CharacterOutput] | None = []
        for edit in edits:
            aligned_tok = process_words(
                text,
                edit,
                reference_transform=tokenizer.transform,
                hypothesis_transform=tokenizer.transform,
            )
            aligned.append(aligned_tok)
            aligned_char.append(process_characters(text, edit))
        spans = cls.get_spans_from_edits(text=text, edits=edits, aligned=aligned_char)
        f_orig_spans = cls._format_spans(spans, span_type=EditSpan, text_type="orig")
        f_edit_spans = cls._format_spans(spans, span_type=EditSpan, text_type="edit")
        tagged = cls.get_tagged_from_spans(texts=[text for _ in edits], spans=f_orig_spans)
        edits_tagged = cls.get_tagged_from_spans(texts=edits, spans=f_edit_spans)
        tokens = cls.get_tokens_from_spans([text for _ in edits], f_orig_spans, tokenizer=tokenizer)
        edits_tokens = cls.get_tokens_from_spans(edits, f_edit_spans, tokenizer=tokenizer)
        return cls(
            text=text,
            spans=spans,
            tagged=tagged,
            tokens=tokens,
            edits=edits,
            edits_tagged=edits_tagged,
            edits_tokens=edits_tokens,
            aligned=aligned,
            aligned_char=aligned_char,
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
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A
                [jiwer transform](https://jitsi.github.io/jiwer/reference/transforms/#transforms)
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        spans = Span.from_list(spans)
        tokens = cls.get_tokens_from_spans(texts=text, spans=spans, tokenizer=tokenizer)[0]
        tagged = cls.get_tagged_from_spans(texts=text, spans=spans)[0]
        return cls(
            text=text,
            spans=spans,
            tagged=tagged,
            tokens=tokens,
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
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A
                [jiwer transform](https://jitsi.github.io/jiwer/reference/transforms/#transforms)
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
        tokens = cls.get_tokens_from_spans(texts=text, spans=spans, tokenizer=tokenizer)[0]
        return cls(
            text=text,
            spans=spans,
            tagged=tagged,
            tokens=tokens,
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
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A
                [jiwer transform](https://jitsi.github.io/jiwer/reference/transforms/#transforms)
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            keep_labels (list[str]):
                Label(s) used to mark selected tokens. If not provided, all tags are kept (Default: []).
            ignore_labels (list[str]):
                Label(s) that are present on tokens but should be ignored while parsing. If not provided, all tags
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
            print(entry)
            >>> Text:
                     Apple Inc. is looking at buying U.K. startup for $1 billion
                Tagged:
                     <ORG>Apple Inc.</ORG> is looking at buying <LOC>U.K.</LOC> startup for <MONEY>$1 billion</MONEY>
                Tokens:
                     Apple Inc. is   looking at   buying U.K. startup for  $1    billion
                     ORG   ORG                           LOC               MONEY MONEY
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
        spans = cls.get_spans_from_tokens(text, labeled_tokens, tokenizer, keep_labels, ignore_labels)
        tagged = cls.get_tagged_from_spans(text, spans=cls._format_spans(spans))[0]
        return cls(
            text=text,
            spans=spans,
            tagged=tagged,
            tokens=labeled_tokens,
            constructor_key=cls.__constructor_key,
        )

    ### Helper Functions ###

    @overload
    @staticmethod
    def _format_spans(
        spans: QESpanInput | Sequence[QESpanInput],
        span_type: type[Span] = ...,
    ) -> list[list[Span]]: ...

    @overload
    @staticmethod
    def _format_spans(
        spans: QESpanWithEditInput | Sequence[QESpanWithEditInput],
        span_type: type[EditSpan] = ...,
        text_type: Literal["orig", "edit"] = ...,
    ) -> list[list[Span]]: ...

    @staticmethod
    def _format_spans(
        spans: QESpanInput | Sequence[QESpanInput] | QESpanWithEditInput | Sequence[QESpanWithEditInput],
        span_type: type[Span | EditSpan] = Span,
        text_type: Literal["orig", "edit"] = "orig",
    ) -> list[list[Span]]:
        """Build a list of spans from a list of `Span` or `EditSpan` objects.

        Args:
            spans (QESpanInput | list[QESpanInput] | QESpanWithEditInput | list[QESpanWithEditInput]): The spans to
                convert to tokens. `SpanListInput` can be a single list of `Span` or `EditSpan`, or a list of
                lists of `Span` or `EditSpan`.
            span_type (type[Span | EditSpan]): The type of span to use for tagging. Default: `Span`.
            text_type (str):
                If `span_type` is `EditSpan`, this specifies whether to use the original text or the edit.

        Returns:
            A list of lists of `Span` objects.
        """
        if len(spans) == 0:
            return [[]]
        if not isinstance(spans[0], list):
            all_spans = [span_type.from_list(spans)]  # type: ignore
        else:
            all_spans = [span_type.from_list(span) for span in spans]  # type: ignore
        if span_type is EditSpan:
            all_spans = [[getattr(span, text_type) for span in span_list] for span_list in all_spans]
        all_spans = cast(list[list[Span]], all_spans)

        # Filter out empty span that could be produced by insertions/deletions on the other text
        all_spans = [[span for span in span_list if span.label is not None] for span_list in all_spans]
        return all_spans

    ### Formatting Methods ###

    @overload
    @staticmethod
    def get_spans_from_edits(text: str, edits: str, aligned: CharacterOutput | None) -> list[EditSpan]: ...

    @overload
    @staticmethod
    def get_spans_from_edits(
        text: str, edits: list[str], aligned: list[CharacterOutput] | None
    ) -> list[list[EditSpan]]: ...

    @staticmethod
    def get_spans_from_edits(
        text: str,
        edits: str | list[str],
        aligned: CharacterOutput | list[CharacterOutput] | None,
        sub_tag: str = "S",
        ins_tag: str = "I",
        del_tag: str = "D",
    ) -> list[EditSpan] | list[list[EditSpan]]:
        """Convert edits to spans over a text and its edits.

        Args:
            text (str): The text.
            edits (str | list[str]): The edited text(s).
            aligned (CharacterOutput | list[CharacterOutput] | None): The character alignment output.
                If `edits` is a single string, this should be a single `CharacterOutput`. If `edits` is a list of
                strings, this should be a list of `CharacterOutput` objects. If not provided, the function will
                obtain it automatically.
            sub_tag (str): The tag for substitutions. Default: "S".
            ins_tag (str): The tag for insertions. Default: "I".
            del_tag (str): The tag for deletions. Default: "D".

        Returns:
            One or more lists of `EditSpan` objects with edit tags.
        """
        if isinstance(edits, str):
            edits = [edits]
        if aligned is None:
            aligned_outputs = [process_characters(text, edit) for edit in edits]
        elif isinstance(aligned, CharacterOutput):
            aligned_outputs = [aligned]
        else:
            aligned_outputs = aligned
        all_spans = []
        for edit, char_aligned_out in zip(edits, aligned_outputs, strict=True):
            edit_spans = []
            for aligned_span in char_aligned_out.alignments[0]:
                params_orig = {
                    "start": aligned_span.ref_start_idx,
                    "end": aligned_span.ref_end_idx,
                    "text": text[aligned_span.ref_start_idx : aligned_span.ref_end_idx],
                }
                params_edit = {
                    "start": aligned_span.hyp_start_idx,
                    "end": aligned_span.hyp_end_idx,
                    "text": edit[aligned_span.hyp_start_idx : aligned_span.hyp_end_idx],
                }
                if aligned_span.type != "equal":
                    if aligned_span.type == "substitute":
                        params_orig["label"] = sub_tag
                        params_edit["label"] = sub_tag
                    elif aligned_span.type == "insert":
                        params_orig["label"] = None
                        params_edit["label"] = ins_tag
                    elif aligned_span.type == "delete":
                        params_orig["label"] = del_tag
                        params_edit["label"] = None
                    span = EditSpan(orig=Span(**params_orig), edit=Span(**params_edit))
                    edit_spans.append(span)
            all_spans.append(edit_spans)
        if len(all_spans) == 1:
            return all_spans[0]
        return all_spans

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
    def get_tokens_from_spans(
        texts: str | list[str],
        spans: list[Span] | list[list[Span]],
        tokenizer: Tokenizer | None = None,
    ) -> ListOfListsOfLabeledToken:
        """Convert spans to tokens.

        Args:
            texts (str | list[str]): The text(s) to which tags should be added.
            spans (list[Span] | list[list[Span]]): The spans to convert to tokens.
            tokenizer (Tokenizer | None): The tokenizer to use for tokenization. If not provided, whitespace
                tokenization is used.

        Returns:
            One or more `LabeledTokenList` representing the tagged texts.
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
            return out
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
        return all_labeled_tokens

    @staticmethod
    def get_text_and_spans_from_tagged(
        tagged: str,
        keep_tags: list[str] = [],
        ignore_tags: list[str] = [],
    ) -> tuple[str, list[Span]]:
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
        span_dicts: list[Span] = []
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
        tokenizer: Tokenizer | None = None,
        keep_labels: list[str] = [],
        ignore_labels: list[str] = [],
    ) -> list[Span]:
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
        """
        tokens = LabeledToken.from_list(tokens)
        if tokenizer is None:
            logger.info("Tokenizer was not provided. Defaulting to whitespace tokenization.")
            tokenizer = WhitespaceTokenizer()
        _, offsets = tokenizer.tokenize_with_offsets(text)
        curr_span_label: str | int | float | None = None
        curr_span_start: int | None = None
        curr_span_end: int | None = None
        spans: list[Span] = []

        # To be considered for a span, a token must have a valid label (not ignored) and a valid character span
        # (not a special token).
        for tok, offset in zip(tokens, offsets[0], strict=True):
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
