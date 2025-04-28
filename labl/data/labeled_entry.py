import re
from collections.abc import Callable, Sequence
from logging import getLogger
from textwrap import dedent, indent
from warnings import warn

import numpy as np
import numpy.typing as npt
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from labl.data.base_entry import BaseLabeledEntry
from labl.data.base_sequence import BaseMultiLabelEntry
from labl.utils.span import Span, SpanList
from labl.utils.token import LabeledToken, LabeledTokenList, LabelType
from labl.utils.tokenizer import Tokenizer, WhitespaceTokenizer, get_tokenizer

logger = getLogger(__name__)


class LabeledEntry(BaseLabeledEntry):
    """Class for a text entry with a set of granular annotations over some of its parts.

    The class provides a centralized object to easily switch between different annotation formats:

    - The original text without labels.
    - A tagged version with <tag>...</tag> labels.
    - A list of spans corresponding to tagged substrings, with start/end indices.
    - A tokenized version of the text with labels associated to each token.

    Attributes:
        text (str):
            The original text.
        spans (list[Span]): A list of lists of `Span` items containing the start/end indices of the span, the
            contained text and a label associated to it.
        tagged (str): A tagged versions of `text` containing tags like `<tag>...</tag>` to mark labels
            from `spans` or `tokens`.
        tokens (list[str]): A list of strings representing the tokenized version of `text`.
        tokens_labels (list[str | int | float | None]): A list of the same length as `tokens` containing labels
            associated with every token. Labels can be strings, numbers or `None`.
        labeled_tokens (list[list[LabeledToken]]): A list of `LabeledToken` objects joining `tokens` and `tokens_labels`.
        tokens_offsets (list[tuple[int, int] | None]): Offsets for each token in `tokens`. The i-th element corresponds
            to the i-th token in `tokens`. The offsets are tuples of the form `(start, end)` corresponding to start and
            end positions of the token in `text`. If the token does not exist in `text`, the offset is `None`.
        label_types (list[type]): A list of the types of labels for the entry.
    """

    # Private constructor key to prevent direct instantiation
    __constructor_key = object()

    def __init__(
        self,
        text: str,
        spans: SpanList,
        tagged: str,
        tokens: list[str],
        tokens_labels: Sequence[LabelType],
        tokens_offsets: list[tuple[int, int] | None],
        constructor_key: object | None = None,
    ):
        """Private constructor for `LabeledEntry`.

        A `LabeledEntry` can be initialized from:

        * A `tagged` text, e.g. `Hello <error>world</error>!`, using `LabeledEntry.from_tagged(tagged=...)`.

        * A `text` and a list of labeled `spans`, e.g. `Hello world!` and `[{'start': 0, 'end': 5, 'label': 'error'}]`,
            using `LabeledEntry.from_spans(text=..., spans=...)`.

        * A list of `labeled_tokens` with string/numeric labels, e.g. `[('Hel', 0.5), ('lo', 0.7), ('world', 1),
            ('!', 0)]`, or two separate lists of `tokens` and `labels` using `LabeledEntry.from_tokens(labeled_tokens=...)`
            or `LabeledEntry.from_tokens(tokens=..., labels=)`.
        """
        if constructor_key != self.__constructor_key:
            raise RuntimeError(
                dedent("""\
                The default constructor for `LabeledEntry` is private. A `LabeledEntry` can be initialized from:

                * A `tagged` text, e.g. `Hello <error>world</error>!`, using `LabeledEntry.from_tagged(tagged=...)`.

                * A `text` and a list of labeled `spans`, e.g. `Hello world!` and `[{'start': 0, 'end': 5, 'label': 'error'}]`,
                    using `LabeledEntry.from_spans(text=..., spans=...)`.

                * A list of `labeled_tokens` with string/numeric labels, e.g. `[('Hel', 0.5), ('lo', 0.7), ('world', 1),
                    ('!', 0)]`, or two separate lists of `tokens` and `labels` using `LabeledEntry.from_tokens(labeled_tokens=...)`
                    or `LabeledEntry.from_tokens(tokens=..., labels=)`.
                """)
            )
        self._text = text
        self._spans = spans
        self._tagged = tagged
        self._tokens = tokens
        self._tokens_labels = tokens_labels
        self._tokens_offsets = tokens_offsets
        self._label_types = self._get_label_types()

    def __str__(self) -> str:
        return dedent(f"""\
          text:
        {indent(self._text, 7 * " ")}
        {self._get_labeled_str()}
        """).strip()

    ### Getters and Setters ###

    @property
    def text(self) -> str:
        """The input text. This is a read-only property."""
        return self._text

    @text.setter
    def text(self, t: str):
        raise RuntimeError("Cannot set the text after initialization")

    @property
    def spans(self) -> SpanList:
        """Labeled spans of the text. This is a read-only property."""
        return self._spans

    @spans.setter
    def spans(self, s: SpanList):
        raise RuntimeError("Cannot set the spans after initialization")

    @property
    def tagged(self) -> str:
        """The tagged version of the text. This is a read-only property."""
        return self._tagged

    @tagged.setter
    def tagged(self, t: str):
        raise RuntimeError("Cannot set the tagged text after initialization")

    @property
    def tokens(self) -> list[str]:
        """The tokenized version of the text. This is a read-only property."""
        return self._tokens

    @tokens.setter
    def tokens(self, t: list[str]):
        raise RuntimeError("Cannot set the tokenized text after initialization")

    @property
    def tokens_labels(self) -> Sequence[LabelType]:
        """The labels associated with the tokens. This is a read-only property."""
        return self._tokens_labels

    @tokens_labels.setter
    def tokens_labels(self, t: Sequence[LabelType]):
        raise RuntimeError("Cannot set token labels after initialization")

    @property
    def tokens_offsets(self) -> list[tuple[int, int] | None]:
        """The offsets for each token in the text. This is a read-only property."""
        return self._tokens_offsets

    @tokens_offsets.setter
    def tokens_offsets(self, t: list[tuple[int, int] | None]):
        raise RuntimeError("Cannot set the tokenized text offsets after initialization")

    @property
    def labeled_tokens(self) -> LabeledTokenList:
        """Returns a list of `LabeledToken` objects joining `tokens` and `tokens_labels` with custom visualization."""
        return LabeledToken.from_list(list(zip(self._tokens, self._tokens_labels, strict=True)))

    @labeled_tokens.setter
    def labeled_tokens(self, t: LabeledTokenList):
        raise RuntimeError("Cannot set the labeled tokens after initialization")

    ### Constructors ###

    @classmethod
    def from_spans(
        cls,
        text: str,
        spans: list[Span] | list[dict[str, LabelType]],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
    ) -> "LabeledEntry":
        """Create a `LabeledEntry` from a text and a list of spans.

        Args:
            text (str):
                The original text.
            spans (list[Span] | list[dict[str, str | int | float | None]]):
                A list or a list of lists of `Span` items or equivalent dicts containing information about specific
                    spans in `text`.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        spans = Span.from_list(spans)
        tokens, tokens_labels, tokens_offsets = cls.get_tokens_from_spans(text=text, spans=spans, tokenizer=tokenizer)
        tagged = cls.get_tagged_from_spans(text=text, spans=spans)
        return cls(
            text=text,
            spans=spans,
            tagged=tagged,
            tokens=tokens,
            tokens_labels=tokens_labels,
            tokens_offsets=tokens_offsets,
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
    ) -> "LabeledEntry":
        """Create a `LabeledEntry` from a tagged text.

        Args:
            tagged (str): Tagged version of `text`.
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
        tokens, tokens_labels, tokens_offsets = cls.get_tokens_from_spans(text=text, spans=spans, tokenizer=tokenizer)
        return cls(
            text=text,
            spans=spans,
            tagged=tagged,
            tokens=tokens,
            tokens_labels=tokens_labels,
            tokens_offsets=tokens_offsets,
            constructor_key=cls.__constructor_key,
        )

    @classmethod
    def from_tokens(
        cls,
        tokens: list[str],
        labels: Sequence[LabelType],
        text: str | None = None,
        offsets: list[tuple[int, int] | None] | None = None,
        keep_labels: list[str] = [],
        ignore_labels: list[str] = [],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
    ) -> "LabeledEntry":
        """Create a `LabeledEntry` from a list of tokens.

        Args:
            tokens (list[str] | None):
                A list of tokens. Can be provided together with `labels` as an alternative to `labeled_tokens`.
            labels (list[str | int | float | None] | None):
                A list of labels for the tokens. Can be provided together with `tokens` as an alternative to
                `labeled_tokens`.
            text (str | None):
                The original text. If not provided, it is detokenized from `tokens` using the tokenizer.
            offsets (list[tuple[int, int] | None] | None):
                The offsets for each token in `tokens`. The i-th element corresponds to the i-th token in `tokens`.
                The offsets are tuples of the form `(start, end)` corresponding to start and end positions of the
                token in `text`. If the token does not exist in `text`, the offset is `None`. If not provided, it is
                computed using the tokenizer.
            tokenizer (str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None): A `Tokenizer`
                used for tokenization. Supports initialization from a `transformers.PreTrainedTokenizer`, and uses
                whitespace tokenization by default.
            keep_labels (list[str]):
                Label(s) used to mark selected tokens. If not provided, all labels are kept (Default: []).
            ignore_labels (list[str]):
                Label(s) that are present on tokens but should be ignored while parsing. If not provided, all labels
                are kept (Default: []).
            tokenizer_kwargs (dict): Additional arguments for the tokenizer.

        Example:
            ```python
            from labl.data.labeled_entry import LabeledEntry

            entry = LabeledEntry.from_tokens(
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
        if len(tokens) != len(labels):
            raise RuntimeError("The length of `tokens` and `labels` must be the same. ")
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        if text is None:
            text = tokenizer.detokenize(tokens)[0]
        if offsets is None:
            _, all_offsets = tokenizer.tokenize_with_offsets(text)
            offsets = all_offsets[0]
        spans = cls.get_spans_from_tokens(text, labels, offsets, tokenizer, keep_labels, ignore_labels)
        tagged = cls.get_tagged_from_spans(text, spans=spans)
        return cls(
            text=text,
            spans=spans,
            tagged=tagged,
            tokens=tokens,
            tokens_labels=labels,
            tokens_offsets=offsets,
            constructor_key=cls.__constructor_key,
        )

    ### Formatting Methods ###

    @staticmethod
    def get_tagged_from_spans(
        text: str,
        spans: list[Span],
    ) -> str:
        """Tags one or more texts using lists of spans.

        Args:
            text (str): The text to which tags should be added.
            spans (list[Span]): The spans to convert to tags.

        Returns:
            The tagged texts.
        """
        if not spans:
            return text
        tagged = text
        sorted_spans = sorted(spans, key=lambda s: s.start)
        offset = 0
        for s in sorted_spans:
            if s.label:
                start = s.start + offset
                end = s.end + offset
                label = s.label
                tagged = f"{tagged[:start]}<{label}>{tagged[start:end]}</{label}>{tagged[end:]}"

                # Update the offset for the next span
                offset += len(str(label)) * 2 + 5  # <{label}>...</{label}>
        return tagged

    @staticmethod
    def get_tokens_from_spans(
        text: str,
        spans: list[Span],
        tokenizer: Tokenizer | None = None,
    ) -> tuple[list[str], Sequence[LabelType], list[tuple[int, int] | None]]:
        """Extracts tokens, labels and offsets from a text and a set of labeled spans.

        Args:
            text (str): The text to which tags should be added.
            spans (list[Span]): The spans to convert to tokens.
            tokenizer (Tokenizer | None): A `Tokenizer` used for text splitting. If not provided, whitespace
                tokenization is used.

        Returns:
            A tuple `(tokens, tokens_labels, tokens_offsets)`, which are three lists of the same length containing
                respectively the tokens, their labels and their (start, end) offsets.
        """
        if tokenizer is None:
            logger.info("Tokenizer was not provided. Defaulting to whitespace tokenization.")
            tokenizer = WhitespaceTokenizer()
        all_tokens, all_tokens_offsets = tokenizer.tokenize_with_offsets(text)
        tokens, tokens_offsets = all_tokens[0], all_tokens_offsets[0]
        if not spans:
            return tokens, [None] * len(tokens), tokens_offsets
        sorted_spans = sorted(spans, key=lambda s: s.start)

        # Pointer for the current position in sorted_spans
        span_idx = 0
        tokens_labels = []

        for offset in tokens_offsets:
            if offset is None:
                tokens_labels.append(None)
                continue
            token_start, token_end = offset
            label = None

            # Skip spans that end before the token starts
            while span_idx < len(sorted_spans) and sorted_spans[span_idx].end <= token_start:
                span_idx += 1

            # Iterate through spans starting from the current span_idx, as long as
            # the span starts before the current token ends. If a span starts
            # at or after the token ends, it (and all subsequent spans) cannot overlap.
            current_check_idx = span_idx
            while current_check_idx < len(sorted_spans) and sorted_spans[span_idx].start < token_end:
                span = sorted_spans[current_check_idx]

                # Check for actual overlap using the standard condition:
                # Does the interval [token_start, token_end) intersect with [span_start, span_end)?
                # Overlap = max(start1, start2) < min(end1, end2)
                if max(token_start, span.start) < min(token_end, span.end):
                    if label is None:
                        label = span.label
                    else:
                        label += span.label  # type: ignore
                current_check_idx += 1  # Move to the next potentially overlapping span
            tokens_labels.append(label)
        return tokens, tokens_labels, tokens_offsets

    @staticmethod
    def get_text_and_spans_from_tagged(
        tagged: str,
        keep_tags: list[str] = [],
        ignore_tags: list[str] = [],
    ) -> tuple[str, SpanList]:
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
        spans = SpanList()
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
                    spans.append(tagged_span)
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
        for span in spans:
            span.text = text_without_tags[span.start : span.end]
        return text_without_tags, spans

    @staticmethod
    def get_spans_from_tokens(
        text: str,
        labels: Sequence[LabelType],
        offsets: list[tuple[int, int] | None] | None = None,
        tokenizer: Tokenizer | None = None,
        keep_labels: list[str] = [],
        ignore_labels: list[str] = [],
    ) -> SpanList:
        """Extract spans and clean text from a list of labeled tokens.

        Args:
            text (str):
                The original text.
            labels (list[str | int | float | None]):
                The labels associated with the tokens.
            offsets (list[tuple[int, int] | None] | None):
                The offsets for each token in `tokens`. The i-th element corresponds to the i-th token in `tokens`.
            tokenizer (Tokenizer | None): The tokenizer to use for
                tokenization. If not provided, whitespace tokenization is used.
            keep_labels (list[str]):
                Token labels that should be ported over to spans. If not provided, all tags are kept (Default: []).
            ignore_labels (list[str]):
                Token labels that should be ignored while parsing. If not provided, all tags are kept (Default: []).

        Returns:
            A list of `Span` objects corresponding to the labeled tokens.
        """
        if offsets is None:
            if tokenizer is None:
                logger.info("Tokenizer was not provided. Defaulting to whitespace tokenization.")
                tokenizer = WhitespaceTokenizer()
            _, all_offsets = tokenizer.tokenize_with_offsets(text)
            offsets = all_offsets[0]
        curr_span_label: LabelType = None
        curr_span_start: int | None = None
        curr_span_end: int | None = None
        spans = SpanList()

        # To be considered for a span, a token must have a valid label (not ignored) and a valid character span
        # (not a special token).
        for label, offset in zip(labels, offsets, strict=True):
            is_ignored = label in ignore_labels
            is_kept = not keep_labels or label in keep_labels
            has_valid_label = is_kept and not is_ignored
            if has_valid_label and offset is not None:
                t_start, t_end = offset
                if label == curr_span_label:
                    curr_span_end = t_end
                else:
                    curr_span_label = label
                    curr_span_start = t_start
                    curr_span_end = t_end
            else:
                curr_span_label = None
                curr_span_start = None
                curr_span_end = None
        if curr_span_label is not None and curr_span_start is not None and curr_span_end is not None:
            spans.append(Span(start=curr_span_start, end=curr_span_end, label=curr_span_label))
        for span in spans:
            span.text = text[span.start : span.end]
        return spans

    ### Utility Functions ###

    def get_tokens(self) -> list[str]:
        return self.tokens

    def get_labels(self) -> Sequence[LabelType]:
        return self.tokens_labels

    ### Helper Functions ###

    def _get_labeled_str(self) -> str:
        tokens_str = str(self.labeled_tokens).replace("\n", "\n" + 8 * " ")
        spans_str = str(self._spans).replace("\n", "\n" + 8 * " ").strip()
        out_str = f"""\
        tagged:
        {indent(self._tagged, 7 * " ")}
        tokens:
        {indent(tokens_str, 7 * " ")}
        spans:
        {indent(spans_str, 7 * " ")}
        """
        return out_str.strip()

    def _get_label_types(self) -> list[type]:
        return list({type(l) for l in self._tokens_labels if l is not None})

    def _relabel_attributes(
        self,
        relabel_fn: Callable[[LabelType], LabelType],
    ) -> None:
        self._tokens_labels = [relabel_fn(label) for label in self._tokens_labels]
        self._spans = SpanList([Span(span.start, span.end, relabel_fn(span.label)) for span in self._spans])
        self._tagged = self.get_tagged_from_spans(self._text, self._spans)

    def _get_labels_array(
        self,
        items: "Sequence[LabeledEntry]",
        dtype: type | None = None,
    ) -> npt.NDArray[np.str_ | np.integer | np.floating]:
        labels_array = np.array([[lab if lab is not None else np.nan for lab in item.tokens_labels] for item in items])
        return labels_array.astype(dtype)


class MultiLabelEntry(BaseMultiLabelEntry[LabeledEntry]):
    """Class for a list of `LabeledEntry` representing multiple labels over the same text."""

    def __str__(self) -> str:
        if len(self) == 0:
            return "No entries available."
        out_str = dedent(f"""\
          text:
        {indent(self[0].text, 7 * " ")}
        """)
        for i, entry in enumerate(self):
            out_str += dedent(f"""\
        === Entry #{i} ===
        {entry._get_labeled_str()}
        """)
        return out_str.strip()
