# Dataset classes for Qualit

from collections.abc import Sequence
from dataclasses import dataclass, field

from jiwer import CharacterOutput, WordOutput, process_characters, process_words
from jiwer.alignment import _construct_comparison_string


@dataclass
class AlignedMixin:
    """Mixin class for aligned output.

    Attributes:
        _aligned_word (WordOutput): The alignment between the machine translation and the post-edit at the word level.
        _aligned_char (CharacterOutput): The alignment between the machine translation and the post-edit at the
            character level.
    """

    _aligned_word: WordOutput | None = field(default=None, kw_only=True)
    _aligned_char: CharacterOutput | None = field(default=None, kw_only=True)

    def __rich__(self):
        return self._get_editing_stats(self._aligned_word, use_rich=True)

    @property
    def word_aligned_output(self):
        """Return the word-aligned output."""
        return self._aligned_word

    @property
    def char_aligned_output(self):
        """Return the character-aligned output."""
        return self._aligned_char

    @property
    def word_aligned_string(self):
        raise NotImplementedError

    @property
    def char_aligned_string(self):
        raise NotImplementedError

    def _get_editing_stats(self, aligned: WordOutput | CharacterOutput, use_rich: bool = False) -> str:
        """Return the editing statistics based on the alignment type using Jiwer."""

        def pluralize(word, count):
            return f"{word}s" if count != 1 else word

        if aligned is None:
            raise ValueError("Alignment is not available.")
        unit = "word" if isinstance(aligned, WordOutput) else "character"
        unit_rate_link = f"https://docs.kolena.com/metrics/wer-cer-mer/#{unit}-error-rate"
        unit_rate = getattr(aligned, "wer" if isinstance(aligned, WordOutput) else "cer")
        metrics_str = ""
        metrics_str += "=== Categories ==="
        metrics_str += f"\nCorrect: {aligned.hits} {pluralize(unit, aligned.hits)}"
        metrics_str += f"\nSubstitutions (S): {aligned.substitutions} {pluralize(unit, aligned.substitutions)}"
        metrics_str += f"\nInsertions (I): {aligned.insertions} {pluralize(unit, aligned.insertions)}"
        metrics_str += f"\nDeletions (D): {aligned.deletions} {pluralize(unit, aligned.deletions)}"
        metrics_str += "\n\n=== Metrics ==="
        if use_rich:
            metrics_str += (
                f"\n[link={unit_rate_link}]{unit.capitalize()} Error Rate ({unit[0].upper()}ER)[/link]: {unit_rate}"
            )
            if isinstance(aligned, WordOutput):
                metrics_str += f"\n[link=https://docs.kolena.com/metrics/wer-cer-mer/#match-error-rate]Match Error Rate (MER)[/link]: {aligned.mer}"
                metrics_str += f"\n[link=https://lightning.ai/docs/torchmetrics/stable/text/word_info_lost.html]Word Information Lost (WIL)[/link]: {aligned.wil}"
                metrics_str += f"\n[link=https://lightning.ai/docs/torchmetrics/stable/text/word_info_preserved.html]Word Information Preserved (WIP)[/link]: {aligned.wip}"
        else:
            metrics_str += f"\n{unit.capitalize()} Error Rate ({unit[0].upper()}ER): {unit_rate}"
            if isinstance(aligned, WordOutput):
                metrics_str += f"\nMatch Error Rate (MER): {aligned.mer}"
                metrics_str += f"\nWord Information Lost (WIL): {aligned.wil}"
                metrics_str += f"\nWord Information Preserved (WIP): {aligned.wip}"
        return metrics_str

    @property
    def word_stats(self):
        """Return the word editing statistics."""
        return self._get_editing_stats(self._aligned_word)

    @property
    def char_stats(self):
        """Return the character editing statistics."""
        return self._get_editing_stats(self._aligned_char)


@dataclass
class MTPEEntry(AlignedMixin):
    """Class for a single entry containing a source text, a machine translation, and a post-edit.

    Attributes:
        source (str): The source text.
        mt (str): The machine translation output.
        pe (str): The post-edited machine translation.
    """

    source: str
    mt: str
    pe: str

    def __str__(self):
        return f"SRC: {self.source}\n\n{self._get_aligned_string(self._aligned_word)}\n{self.word_stats}"

    def __post_init__(self):
        if self._aligned_word is None:
            self._aligned_word = process_words(self.mt, self.pe)
        if self._aligned_char is None:
            self._aligned_char = process_characters(self.mt, self.pe)

    def _get_aligned_string(self, aligned: WordOutput | CharacterOutput):
        """Return the word-aligned output string based on the alignment type using Jiwer."""
        if aligned is None:
            raise ValueError("Alignment is not available.")
        string = _construct_comparison_string(
            aligned.references[0],
            aligned.hypotheses[0],
            aligned.alignments[0],
            include_space_seperator=not isinstance(aligned, CharacterOutput),
        )
        return string.replace("REF:", " MT:", 1).replace("HYP:", " PE:", 1)

    @property
    def word_aligned_string(self):
        """Return the word-aligned entry string."""
        return f"SRC: {self.source}\n\n{self._get_aligned_string(self._aligned_word)}"

    @property
    def char_aligned_string(self):
        """Return the character-aligned entry string."""
        return f"SRC: {self.source}\n\n{self._get_aligned_string(self._aligned_char)}"


@dataclass
class MTPEDataset(AlignedMixin, Sequence):
    """Dataset class for MT data with post-edits.

    Attributes:
        entries (list[MTPEEntry]): A list of MTPEEntry objects.
    """

    entries: list[MTPEEntry]

    def __str__(self):
        return self.word_aligned_string + f"\n{self.word_stats}"

    def __getitem__(self, idx):
        return self.entries[idx]

    def __len__(self):
        return len(self.entries)

    def __add__(self, other: "MTPEDataset") -> "MTPEDataset":
        return MTPEDataset(self.entries + other.entries)

    @property
    def word_aligned_string(self):
        """Return the word-aligned dataset string."""
        return "\n".join(f"#{idx}\n{entry.word_aligned_string}" for idx, entry in enumerate(self.entries, start=1))

    @property
    def char_aligned_string(self):
        """Return the character-aligned dataset string."""
        return "\n".join(f"#{idx}\n{entry.char_aligned_string}" for idx, entry in enumerate(self.entries, start=1))

    @staticmethod
    def from_sentences(source: list[str], mt: list[str], pe: list[str]) -> "MTPEDataset":
        """Load a dataset from lists of source, machine translation, and post-edit sentences."""
        entries = [MTPEEntry(src, m, p) for src, m, p in zip(source, mt, pe, strict=False)]
        _aligned_word = process_words(mt, pe)
        _aligned_char = process_characters(mt, pe)
        return MTPEDataset(entries, _aligned_word=_aligned_word, _aligned_char=_aligned_char)
