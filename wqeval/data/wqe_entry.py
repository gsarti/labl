from dataclasses import dataclass
from typing import Literal

from jiwer import process_characters, process_words

from .aligned_mixin import AlignedSequencesMixin, SequenceAlignmentType, regex_wordsplit


@dataclass
class WordLevelQEEntry(AlignedSequencesMixin):
    """Class for a single text entry with word-level quality estimation utilities.

    Attributes:
        source (str): The source text.
        mt (str): The machine translation output.
        pe (str): The post-edited machine translation.
    """

    mt: str
    pe: str | None = None
    src: str | None = None

    def __str__(self):
        return self.word_aligned_string

    def __post_init__(self):
        if self._aligned_word is None and self.pe is not None:
            self._aligned_word = process_words(
                self.mt, self.pe, reference_transform=regex_wordsplit, hypothesis_transform=regex_wordsplit
            )
        if self._aligned_char is None and self.pe is not None:
            self._aligned_char = process_characters(self.mt, self.pe)

    def _get_aligned_string(
        self, alignment_type: Literal[SequenceAlignmentType.WORD, SequenceAlignmentType.CHAR], add_stats: bool = False
    ) -> str:
        out = ""
        aligned = self._aligned_word if alignment_type == SequenceAlignmentType.WORD else self._aligned_char
        if self.src is not None:
            out += f"SRC: {self.src}\n\n"
            if aligned is not None:
                aligned_str = self._get_jiwer_string(aligned)
                aligned_str = aligned_str.replace("REF:", " MT:", 1).replace("HYP:", " PE:", 1)
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
