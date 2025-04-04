from jiwer import AbstractTransform, CharacterOutput, Compose, WordOutput, process_characters, process_words
from jiwer.alignment import _construct_comparison_string


class AlignedSequencesMixin:
    """Base class for handling sequences alignment using the [`jiwer`](https://jitsi.github.io/jiwer/) library.

    Args:
        text (str): The original text.
        edits (list[str]): A list of one or more edited strings.
        transform (AbstractTransform | Compose): A transform or compose object to apply to the text.
    """

    def __init__(self, text: str, edits: list[str], transform: AbstractTransform | Compose) -> None:
        self._aligned: list[WordOutput] | None = []
        self._aligned_char: list[CharacterOutput] | None = []
        for edit in edits:
            aligned_tok = process_words(
                text,
                edit,
                reference_transform=transform,
                hypothesis_transform=transform,
            )
            self._aligned.append(aligned_tok)
            aligned_char = process_characters(text, edit)
            self._aligned_char.append(aligned_char)

    def __rich__(self):
        if self._aligned is None:
            raise ValueError("Alignment is not available.")
        return "\n\n".join(self._get_editing_stats(a, use_rich=True) for a in self._aligned)

    @property
    def aligned(self) -> list[WordOutput] | None:
        """Returns a list of [`jiwer.WordOutput`](https://jitsi.github.io/jiwer/reference/process/#process.WordOutput)
        aligned at the token level.
        """
        return self._aligned

    @property
    def aligned_char(self) -> list[CharacterOutput] | None:
        """Returns a list of [`jiwer.CharacterOutput`](https://jitsi.github.io/jiwer/reference/process/#process.CharacterOutput)
        aligned at the token level.
        """
        return self._aligned_char

    @property
    def aligned_str(self) -> str:
        """Returns the aligned string at the token level with [`jiwer.visualize_alignment`](https://jitsi.github.io/jiwer/reference/alignment/#alignment.visualize_alignment)."""
        if self._aligned is None:
            raise RuntimeError("Token-level alignment is not available.")
        return self._get_aligned_string(self._aligned)

    @property
    def aligned_char_str(self) -> str:
        """Returns the aligned string at the character level with [`jiwer.visualize_alignment`](https://jitsi.github.io/jiwer/reference/alignment/#alignment.visualize_alignment)."""
        if self._aligned_char is None:
            raise RuntimeError("Character-level alignment is not available.")
        return self._get_aligned_string(self._aligned_char)

    def _get_aligned_string(
        self, al_outputs: list[WordOutput] | list[CharacterOutput], add_stats: bool = False
    ) -> str:
        if al_outputs is None:
            raise RuntimeError("Alignment is not available.")
        out = ""
        if self._aligned is not None:
            for aligned in al_outputs:
                aligned_str = self._get_jiwer_string(aligned)
                aligned_str = aligned_str.replace("REF:", " TEXT:", 1).replace("HYP:", " EDIT:", 1)
                lines = aligned_str.split("\n")
                lines[2] = " " + lines[2]
                aligned_str = "\n".join(lines)
                out += f"{aligned_str}"
                if add_stats:
                    out += f"\n{self._get_editing_stats(aligned)}"
                out += "\n\n"
        return out

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
            include_space_seperator=isinstance(aligned, WordOutput),
        )
