from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from jiwer import process_characters, process_words

from .aligned_mixin import AlignedSequencesMixin, SequenceAlignmentType, regex_wordsplit
from .wqe_entry import WordLevelQEEntry


@dataclass
class WordLevelQEDataset(AlignedSequencesMixin):
    """Dataset class for word-level QE entries.

    Attributes:
        data (list[WordLevelQEEntry]): A list of WordLevelQEEntry objects.
    """

    data: list[WordLevelQEEntry]

    def __str__(self) -> str:
        return self.word_aligned_string + f"\n{self.get_stats(self._aligned_word)}"

    def __getitem__(self, idx) -> WordLevelQEEntry:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    def __add__(self, other: "WordLevelQEDataset") -> "WordLevelQEDataset":
        return WordLevelQEDataset(self.data + other.data)

    def _get_aligned_string(
        self, alignment_type: Literal[SequenceAlignmentType.WORD, SequenceAlignmentType.CHAR]
    ) -> str:
        return "\n".join(
            f"#{idx}\n{entry._get_aligned_string(alignment_type)}" for idx, entry in enumerate(self.data, start=1)
        )

    @property
    def word_aligned_string(self) -> str:
        """Return the word-aligned dataset string."""
        return self._get_aligned_string(SequenceAlignmentType.WORD)

    @property
    def char_aligned_string(self) -> str:
        """Return the character-aligned dataset string."""
        return self._get_aligned_string(SequenceAlignmentType.CHAR)

    @staticmethod
    def from_sentences(
        orig: str | Sequence[str],
        edit: str | Sequence[str] | None = None,
    ) -> "WordLevelQEDataset":
        """Load a dataset from lists of machine translations, with optional source and post-edit sentences."""
        orig = list(orig)
        edit = list(edit) if edit is not None else None
        _aligned_word = None
        _aligned_char = None
        if edit is not None:
            _aligned_word = process_words(
                orig, edit, reference_transform=regex_wordsplit, hypothesis_transform=regex_wordsplit
            )
            _aligned_char = process_characters(orig, edit)
        data_list = []
        for idx, orig_i in enumerate(orig):
            edit_i = None
            if edit is not None and idx < len(edit):
                edit_i = edit[idx]
            data_list.append(WordLevelQEEntry(orig=orig_i, edit=edit_i))
        return WordLevelQEDataset(data_list, _aligned_word=_aligned_word, _aligned_char=_aligned_char)
