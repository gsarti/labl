import re
from dataclasses import dataclass, field
from enum import Enum

from jiwer import CharacterOutput, WordOutput
from jiwer.alignment import _construct_comparison_string
from jiwer.transforms import AbstractTransform, Compose, ReduceToListOfListOfWords, Strip


class SequenceAlignmentType(Enum):
    """Enum for alignment types."""

    WORD = "Word"
    CHAR = "Character"


@dataclass
class AlignedSequencesMixin:
    """Mixin class for handling aligned sequences.

    Attributes:
        _aligned_word (jiwer.WordOutput): Aligned output at the word level.
        _aligned_char (jiwer.CharacterOutput): Aligned output at the character level.
    """

    _aligned_word: WordOutput | None = field(default=None, kw_only=True)
    _aligned_char: CharacterOutput | None = field(default=None, kw_only=True)

    def __rich__(self):
        return self._get_editing_stats(self._aligned_word, use_rich=True)

    @property
    def word_aligned_output(self) -> WordOutput | None:
        """Return the word-aligned output."""
        return self._aligned_word

    @property
    def char_aligned_output(self) -> CharacterOutput | None:
        """Return the character-aligned output."""
        return self._aligned_char

    @property
    def word_aligned_string(self) -> str:
        raise NotImplementedError

    @property
    def char_aligned_string(self) -> str:
        raise NotImplementedError

    def get_stats(self, aligned: WordOutput | CharacterOutput | None) -> str:
        """Return aligned strings statistics."""
        return self._get_editing_stats(aligned)

    @property
    def word_stats(self) -> str:
        """Return the word-aligned statistics."""
        return self.get_stats(self._aligned_word)

    @property
    def char_stats(self) -> str:
        """Return the character-aligned statistics."""
        return self.get_stats(self._aligned_char)

    def _get_editing_stats(self, aligned: WordOutput | CharacterOutput | None, use_rich: bool = False) -> str:
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

    def _get_jiwer_string(self, aligned: WordOutput | CharacterOutput | None) -> str:
        """Return the word-aligned output string based on the alignment type using Jiwer."""
        if aligned is None:
            raise ValueError("Alignment is not available.")
        return _construct_comparison_string(
            aligned.references[0],
            aligned.hypotheses[0],
            aligned.alignments[0],
            include_space_seperator=not isinstance(aligned, CharacterOutput),
        )


class RegexReduceToListOfListOfWords(AbstractTransform):
    """
    Version of `ReduceToListOfWords` using Regex for splitting.
    """

    def __init__(self, exp: str = r"[\w']+|[.,!?:;'”#$%&\(\)\*\+-/<=>@\[\]^_`{|}~\"]"):
        """
        Args:
            exp: the Regex expression to use for splitting. Default: r"[\\w']+|[.,!?:;'”#$%&\\(\\)\\*\\+-/<=>@\\[\\]^_`{|}~\"]"
        """
        self.exp = exp

    def process_string(self, s: str):
        return [[word for word in re.findall(self.exp, s) if len(word) >= 1]]

    def process_list(self, inp: list[str]):
        sentence_collection = []
        for sentence in inp:
            list_of_words = self.process_string(sentence)[0]
            sentence_collection.append(list_of_words)
        if len(sentence_collection) == 0:
            return [[]]
        return sentence_collection


orig_wordsplit = Compose([Strip(), ReduceToListOfListOfWords()])
regex_wordsplit = Compose([Strip(), RegexReduceToListOfListOfWords()])
