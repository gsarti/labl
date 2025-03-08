import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from warnings import warn

from jiwer import process_characters, process_words

from .aligned_mixin import AlignedSequencesMixin, SequenceAlignmentType, regex_wordsplit


@dataclass
class WordLevelQEEntry(AlignedSequencesMixin):
    """Class for a single text entry with word-level quality estimation utilities.

    Attributes:
        orig (str): The original text.
        edit (str): The edited text.
    """

    orig: str
    edit: str | None = None
    tagged_spans: list[dict[str, str | int]] | None = None

    def __str__(self):
        return self.word_aligned_string

    def __post_init__(self):
        if self._aligned_word is None and self.edit is not None:
            self._aligned_word = process_words(
                self.orig, self.edit, reference_transform=regex_wordsplit, hypothesis_transform=regex_wordsplit
            )
        if self._aligned_char is None and self.edit is not None:
            self._aligned_char = process_characters(self.orig, self.edit)

    def _get_aligned_string(
        self, alignment_type: Literal[SequenceAlignmentType.WORD, SequenceAlignmentType.CHAR], add_stats: bool = False
    ) -> str:
        out = ""
        aligned = self._aligned_word if alignment_type == SequenceAlignmentType.WORD else self._aligned_char
        if aligned is not None:
            aligned_str = self._get_jiwer_string(aligned)
            aligned_str = aligned_str.replace("REF:", " ORIG.:", 1).replace("HYP:", " EDIT.:", 1)
            out += f"{aligned_str}"
            if add_stats:
                out += f"\n{self.get_stats(aligned)}"
        return out

    @property
    def word_aligned_string(self) -> str:
        """Return the word-aligned entry string."""
        return self._get_aligned_string(SequenceAlignmentType.WORD)

    @property
    def char_aligned_string(self) -> str:
        """Return the character-aligned entry string."""
        return self._get_aligned_string(SequenceAlignmentType.CHAR)

    @classmethod
    def from_tagged_text(
        cls,
        text: str,
        keep_tag_types: Sequence[str] = ["h"],
        ignore_tag_types: Sequence[str] = [],
    ) -> "WordLevelQEEntry":
        """Parse a text containing tags into a WordLevelQEEntry instance.

        Args:
            text (str):
                Original text containing tags that needs to be parsed.
            keep_tag_types (Sequence[str]):
                Tag(s) used to mark selected spans, e.g. `<h>...</h>`, `<error>...</error>`.
                Default: 'h'.
            ignore_tag_types (Sequence[str]):
                Tag(s) that are present in the text but should be ignored while parsing. Default: [].
        Returns:
            An initialized `WordLevelQEEntry` instance.
        """
        tag_regex = re.compile(rf"<\/?(?:{'|'.join(list(set(keep_tag_types) | set(ignore_tag_types)))})>")
        any_tag_regex = re.compile(r"<\/?(?:\w+)>")
        if not text:
            raise RuntimeError("Text cannot be empty")
        if not keep_tag_types:
            raise RuntimeError("At least one tag type must be provided in keep_tag_types")

        text_without_tags = ""
        tagged_spans = []
        current_pos = 0
        open_tags = []
        open_positions = []

        for match in tag_regex.finditer(text):
            match_text = match.group(0)
            start, end = match.span()

            # Add text before the tag
            text_without_tags += text[current_pos:start]
            current_pos = end

            # Check if opening or closing tag
            if match_text.startswith("</"):
                tag_name = match_text[2:-1]
                if not open_tags or open_tags[-1] != tag_name:
                    raise RuntimeError(f"Closing tag {match_text} without matching opening tag")

                # Create span for the highlighted text
                open_pos = open_positions.pop()
                open_tag = open_tags.pop()
                if tag_name not in ignore_tag_types:
                    tagged_span = {
                        "start": open_pos,
                        "end": len(text_without_tags),
                        "tag": open_tag,
                    }
                    tagged_spans.append(tagged_span)
            else:
                # Opening tag
                tag_name = match_text[1:-1]
                if tag_name not in keep_tag_types and tag_name not in ignore_tag_types:
                    raise RuntimeError(
                        f"Unexpected tag type: {tag_name}. "
                        "Specify tag types that should be preserved in the `keep_tag_types` argument, "
                        "and those that should be ignored in the `ignore_tag_types` argument."
                    )

                open_tags.append(tag_name)
                open_positions.append(len(text_without_tags))

        # Add remaining text
        text_without_tags += text[current_pos:]

        # Check for unclosed tags
        if open_tags:
            raise RuntimeError(f"Unclosed tags: {', '.join(open_tags)}")

        # If the text contains a tag that was neither kept nor ignored, raise a warning
        unexpected_tags = any_tag_regex.search(text_without_tags)
        if unexpected_tags:
            warn(
                "The text contains tag types that were not specified in keep_tag_types or ignore_tag_types: "
                f"{unexpected_tags.group(0)}. These tags are now preserved in the output. If these should ignored "
                "instead, add them to the `ignore_tag_types` argument.",
                stacklevel=2,
            )

        for span in tagged_spans:
            span["text"] = text_without_tags[span["start"] : span["end"]]

        # Create and return the WordLevelQEEntry
        return cls(orig=text_without_tags, tagged_spans=tagged_spans)
