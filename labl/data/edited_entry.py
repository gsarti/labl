from collections.abc import Callable, Sequence
from logging import getLogger
from textwrap import dedent, indent

import numpy as np
import numpy.typing as npt
from jiwer import WordOutput
from jiwer.alignment import _construct_comparison_string
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from labl.data.base_entry import BaseLabeledEntry
from labl.data.base_sequence import BaseMultiLabelEntry
from labl.data.labeled_entry import LabeledEntry
from labl.utils.jiwer_ext import process_words
from labl.utils.tokenizer import Tokenizer, WhitespaceTokenizer, get_tokenizer
from labl.utils.typing import LabelType

logger = getLogger(__name__)


class EditedEntry(BaseLabeledEntry):
    """Class for a pair of text entries (`orig` and `edit`) where word-level annotations are obtained from the aligned
        tokens of the two entries.

    Attributes:
        orig (LabeledEntry): The original entry.
        edit (LabeledEntry): The edited entry.
        aligned (WordOutput): A `jiwer.WordOutput` with aligned tokens for `orig` and `edit`, using tokenized the
            provided tokenizer.
        has_gaps (bool): Whether the token sequence has gaps. Gaps are used for text/edit pairs to mark the
            positions of insertions and deletions in the original/edited texts, respectively. If `False`, it means gap
            annotations were merged to the next token to the right.
    """

    # Private constructor key to prevent direct instantiation
    __constructor_key = object()

    def __init__(
        self,
        orig: LabeledEntry,
        edit: LabeledEntry,
        aligned: WordOutput,
        has_gaps: bool,
        has_bos_token: bool,
        has_eos_token: bool,
        constructor_key: object | None = None,
    ):
        """Private constructor for `EditedEntry`.

        One or more `EditedEntry` can be initialized from a  `text` and one or more `edits`, e.g. `Hello world!` and
            `["Goodbye world!", "Hello planet!"]`, using `EditedEntry.from_edits(text=..., edits=...)`.
        """
        if constructor_key != self.__constructor_key:
            raise RuntimeError(
                dedent("""\
                The default constructor for `EditedEntry` is private. One or more `EditedEntry` can be initialized from
                a  `text` and one or more `edits`, e.g. `Hello world!` and `["Goodbye world!", "Hello planet!"]`, using
                 `EditedEntry.from_edits(text=..., edits=...)`.
                """)
            )
        self._orig = orig
        self._edit = edit
        self._aligned = aligned
        self._has_gaps = has_gaps
        self._has_bos_token = has_bos_token
        self._has_eos_token = has_eos_token
        self._label_types = list(set(self._orig._label_types) | set(self._edit._label_types))

    def __str__(self) -> str:
        return dedent(f"""\
          orig.text:
        {indent(self._orig._text, 12 * " ")}
        {self._get_edit_str()}
        """).strip()

    ### Getters and Setters ###

    @property
    def orig(self) -> LabeledEntry:
        """The `LabeledEntry` for the original text."""
        return self._orig

    @orig.setter
    def orig(self, t: LabeledEntry):
        raise RuntimeError("Cannot set original entry.")

    @property
    def edit(self) -> LabeledEntry:
        """The `LabeledEntry` entry for the edited text."""
        return self._edit

    @edit.setter
    def edit(self, t: LabeledEntry):
        raise RuntimeError("Cannot set edited entry.")

    @property
    def aligned(self) -> WordOutput:
        """Aligned output using `jiwer` for `orig` and `edit`."""
        return self._aligned

    @aligned.setter
    def aligned(self, t: WordOutput):
        raise RuntimeError("Cannot set aligned tokens.")

    @property
    def has_gaps(self) -> bool:
        """Boolean flag marking whether the token sequence has added gaps for insertion/deletion annotations."""
        return self._has_gaps

    @has_gaps.setter
    def has_gaps(self, t: bool):
        raise RuntimeError("Cannot set gaps.")

    @property
    def has_bos_token(self) -> bool:
        """Boolean flag marking whether the tokenizer has a beginning-of-sequence token."""
        return self._has_bos_token

    @has_bos_token.setter
    def has_bos_token(self, t: bool):
        raise RuntimeError("Cannot set beginning-of-sequence token.")

    @property
    def has_eos_token(self) -> bool:
        """Boolean flag marking whether the tokenizer has an end-of-sequence token."""
        return self._has_eos_token

    @has_eos_token.setter
    def has_eos_token(self, t: bool):
        raise RuntimeError("Cannot set end-of-sequence token.")

    @property
    def aligned_str(self) -> str:
        """Aligned string at the token level with [`jiwer.visualize_alignment`](https://jitsi.github.io/jiwer/reference/alignment/#alignment.visualize_alignment)."""
        aligned_str_out = ""
        aligned_str = _construct_comparison_string(
            self._aligned.references[0],
            self._aligned.hypotheses[0],
            self._aligned.alignments[0],
            include_space_seperator=True,
        )
        aligned_str = aligned_str.replace("REF:", "ORIG:", 1).replace("HYP:", "EDIT:", 1)
        lines = aligned_str.split("\n")
        lines[2] = " " + lines[2]
        aligned_str = "\n".join(lines)
        aligned_str_out += f"{aligned_str}"
        return aligned_str_out

    @aligned_str.setter
    def aligned_str(self, t: str):
        raise RuntimeError("Cannot set string representation of aligned output.")

    ### Constructors ###

    @classmethod
    def from_edits(
        cls,
        text: str,
        edits: str | list[str],
        tokenizer: str | Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        tokenizer_kwargs: dict = {},
        with_gaps: bool = True,
        keep_final_gap: bool = True,
        sub_label: str = "S",
        ins_label: str = "I",
        del_label: str = "D",
        gap_token: str = "▁",
    ) -> "EditedEntry | MultiEditEntry":
        """Create a `EditedEntry` or an `MultiEditEntry` from a text and one or more edits.

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
            keep_final_gap (bool): Whether to keep the final gap token. Default: True.
            sub_label (str): The label for substitutions. Default: "S".
            ins_label (str): The label for insertions. Default: "I".
            del_label (str): The label for deletions. Default: "D".
            gap_token (str): The token to use for gaps. Default: "▁".

        Returns:
            A single `EditedEntry` if `edits` is a single string, otherwise an `MultiEditEntry` with one entry per
                edit.

        Example:
            ```python
            from labl.data.edited_entry import EditedEntry

            entries = EditedEntry.from_edits(
                text="a simple example",
                edits=["this is a simple enough test, you know?", "an example"],
                tokenizer="facebook/nllb-200-3.3B",
                tokenizer_kwargs={
                    "tgt_lang": "ita_Latn",
                    "add_special_tokens": True,
                },
            )
            print(entries[0].aligned_str)
            >>> ORIG: ita_Latn ***** *** ▁a ▁simple ******* ***** * **** ***** ▁example </s>
                EDIT: ita_Latn ▁this ▁is ▁a ▁simple ▁enough ▁test , ▁you ▁know        ? </s>
                                   I   I                  I     I I    I     I        S
            ```
        """
        tokenizer = get_tokenizer(tokenizer, tokenizer_kwargs)
        edits = [edits] if isinstance(edits, str) else edits
        tokens, offsets = tokenizer.tokenize_with_offsets(text)
        tokens_with_gaps, offsets_with_gaps = tokenizer._add_gaps_to_tokens_and_offsets(tokens, offsets, gap_token)
        tokens, offsets = tokens[0], offsets[0]
        tokens_with_gaps, offsets_with_gaps = tokens_with_gaps[0], offsets_with_gaps[0]
        all_edits_tokens, all_edits_offsets = tokenizer.tokenize_with_offsets(edits)
        all_edits_tokens_with_gaps, all_edits_offsets_with_gaps = tokenizer._add_gaps_to_tokens_and_offsets(
            all_edits_tokens, all_edits_offsets, gap_token=gap_token
        )
        entries = MultiEditEntry()
        for edit, e_tokens, e_offsets, e_tokens_with_gaps, e_offsets_with_gaps in zip(
            edits,
            all_edits_tokens,
            all_edits_offsets,
            all_edits_tokens_with_gaps,
            all_edits_offsets_with_gaps,
            strict=True,
        ):
            aligned = process_words(
                texts=[tokens], edits=[e_tokens], is_text_pre_transformed=True, is_edit_pre_transformed=True
            )
            tokens_labels, e_tokens_labels = cls.get_tokens_labels_from_edit(
                text=text,
                edit=edit,
                tokens=tokens_with_gaps,
                tokens_offsets=offsets_with_gaps,
                edit_tokens=e_tokens_with_gaps,
                edit_tokens_offsets=e_offsets_with_gaps,
                aligned=aligned,
                tokenizer=tokenizer,
                sub_label=sub_label,
                ins_label=ins_label,
                del_label=del_label,
                gap_token=gap_token,
            )
            if with_gaps:
                out_tokens = tokens_with_gaps
                out_offsets = offsets_with_gaps
                out_edit_tokens = e_tokens_with_gaps
                out_edit_offsets = e_offsets_with_gaps
            else:
                # If an ad-hoc EOS is added, it is always kept
                if tokenizer.has_eos_token:
                    if not keep_final_gap:
                        raise RuntimeError(
                            "The tokenizer has an EOS token, but `keep_final_gap` is set to False."
                            "The EOS token will be kept."
                        )
                tokens_labels = tokenizer._merge_gap_annotations(
                    [tokens_labels], has_bos_token=tokenizer.has_bos_token, keep_final_gap=keep_final_gap
                )[0]
                e_tokens_labels = tokenizer._merge_gap_annotations(
                    [e_tokens_labels], has_bos_token=tokenizer.has_bos_token, keep_final_gap=keep_final_gap
                )[0]

                # If gaps are merged, the last gap is kept regardless of it being a gap or not to mark end-insertions.
                # If the tokenizer did not have an EOS token for that, the sequence will have an extra token and offsets
                # will need to be adjusted.
                if not keep_final_gap:
                    out_tokens, out_offsets = tokens, offsets
                    out_edit_tokens, out_edit_offsets = e_tokens, e_offsets
                else:
                    out_tokens, out_offsets = tokens + [gap_token], offsets + [None]
                    out_edit_tokens, out_edit_offsets = e_tokens + [gap_token], e_offsets + [None]
            entry = EditedEntry(
                orig=LabeledEntry.from_tokens(
                    tokens=out_tokens,
                    labels=tokens_labels,
                    text=text,
                    offsets=out_offsets,
                    tokenizer=tokenizer,
                ),
                edit=LabeledEntry.from_tokens(
                    tokens=out_edit_tokens,
                    labels=e_tokens_labels,
                    text=edit,
                    offsets=out_edit_offsets,
                    tokenizer=tokenizer,
                ),
                aligned=aligned,
                has_gaps=with_gaps,
                has_bos_token=tokenizer.has_bos_token,
                has_eos_token=tokenizer.has_eos_token,
                constructor_key=cls.__constructor_key,
            )
            entries.append(entry)
        if len(entries) == 1:
            return entries[0]
        entries._label_types = entries._get_label_types()
        return entries

    ### Formatting Methods ###

    @classmethod
    def get_tokens_labels_from_edit(
        cls,
        text: str,
        edit: str,
        tokens: list[str] | None = None,
        tokens_offsets: list[tuple[int, int] | None] | None = None,
        edit_tokens: list[str] | None = None,
        edit_tokens_offsets: list[tuple[int, int] | None] | None = None,
        aligned: WordOutput | None = None,
        tokenizer: Tokenizer | None = None,
        sub_label: str = "S",
        ins_label: str = "I",
        del_label: str = "D",
        gap_token: str = "▁",
    ) -> tuple[Sequence[str | None], Sequence[str | None]]:
        """Convert text edits to token labels marking insertions, deletions and substitutions. The returned labels
        include gaps before/after each token, which can be merged to the right to match the original token sequence.

        Args:
            text (str): The original text.
            edit (str): The edited text.
            tokens (list[str] | None): The tokenized version of `text`. If not provided, it will be computed using
                `tokenzier`. Default: `None`.
            tokens_offsets (list[tuple[int, int] | None]): The offsets of `tokens` in `text`. If not provided, it will
                be computed using `tokenzier`. Default: `None`.
            edit_tokens (list[str] | None): The tokenized version of `edit`. If not provided, it will be computed using
                `tokenzier`. Default: `None`.
            edit_tokens_offsets (list[tuple[int, int] | None]): The offsets of `edit_tokens` in `edit`. If not
                provided, it will be computed using `tokenzier`. Default: `None`.
            aligned (WordOutput | None): The aligned `WordOutput` between `text` and `edit`. If not provided, it will
                be obtained automatically using `tokenizer` for spltting. Default: `None`.
            tokenizer (Tokenizer | None): A `Tokenizer` used for text splitting. If not provided, whitespace
                tokenization is used by default. Default: `None`.
            sub_label (str): The label for substitutions. Default: "S".
            ins_label (str): The label for insertions. Default: "I".
            del_label (str): The label for deletions. Default: "D".
            gap_token (str): The token to use for gaps. Default: "▁".

        Returns:
            A tuple containing two lists of labels (one for `text`, one for `edit`). Each label can be either None
                (if the token was not edited) or one of `sub_label`, `ins_label` or `del_label` depending on the type
                of operation associated with the token.
        """
        if tokenizer is None:
            logger.info("Tokenizer was not provided. Defaulting to whitespace tokenization.")
            tokenizer = WhitespaceTokenizer()
        if aligned is None:
            aligned = process_words(
                text, edit, texts_transform=tokenizer.transform, edits_transform=tokenizer.transform
            )
        tokens, tokens_offsets = cls._get_tokens_with_gaps(tokenizer, text, tokens, tokens_offsets, gap_token)
        edit_tokens, edit_tokens_offsets = cls._get_tokens_with_gaps(
            tokenizer, edit, edit_tokens, edit_tokens_offsets, gap_token
        )
        tokens_labels: list[str | None] = [None] * len(tokens)
        edit_tokens_labels: list[str | None] = [None] * len(edit_tokens)
        for alignment in aligned.alignments[0]:
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
                tokens_labels[text_start_idx * 2] = ins_label
            elif alignment.type in ("delete", "substitute"):
                label = sub_label if alignment.type == "substitute" else del_label
                for idx in range(text_start_idx, text_end_idx):
                    tokens_labels[idx * 2 + 1] = label
            if alignment.type == "delete":
                edit_tokens_labels[edit_start_idx * 2] = del_label
            elif alignment.type in ("insert", "substitute"):
                label = sub_label if alignment.type == "substitute" else ins_label
                for idx in range(edit_start_idx, edit_end_idx):
                    edit_tokens_labels[idx * 2 + 1] = label
        return tokens_labels, edit_tokens_labels

    ### Utility Methods ###

    def get_tokens(self) -> list[str]:
        return self.orig.tokens

    def get_labels(self) -> Sequence[LabelType]:
        return self.orig.tokens_labels

    def merge_gap_annotations(
        self,
        merge_fn: Callable[[Sequence[LabelType]], LabelType] | None = None,
        keep_final_gap: bool = True,
    ) -> None:
        """Merge gap annotations in the tokens of `orig` and `edit`.

        This method is equivalent to calling `EditedEntry.from_edits` with `with_gaps=False`. Gap annotations are merged
        to the next non-gap token to the right, and the gap label is added to the label of the non-gap token. The last
        gap is kept to account for insertions at the end of the text.

        E.g. `GAP Hello GAP World GAP ! GAP` becomes `Hello World ! GAP`.
             `  I     S   I               I`         `   IS     I     I`
        """
        if not self._has_gaps:
            raise RuntimeError("Gaps for the current entry were already merged.")
        has_bos = self._has_bos_token
        if self.has_eos_token:
            if not keep_final_gap:
                raise RuntimeError(
                    "The tokenizer has an EOS token, but `keep_final_gap` is set to False. The EOS token will be kept."
                )
        o_tok, o_lab, o_off = self._orig._tokens, self._orig._tokens_labels, self._orig._tokens_offsets
        e_tok, e_lab, e_off = self._edit._tokens, self._edit._tokens_labels, self._edit._tokens_offsets
        self._orig._tokens = Tokenizer._remove_gap_tokens([o_tok], self._has_bos_token, keep_final_gap)[0]
        self._edit._tokens = Tokenizer._remove_gap_tokens([e_tok], self._has_bos_token, keep_final_gap)[0]
        self._orig._tokens_labels = Tokenizer._merge_gap_annotations([o_lab], merge_fn, has_bos, keep_final_gap)[0]
        self._edit._tokens_labels = Tokenizer._merge_gap_annotations([e_lab], merge_fn, has_bos, keep_final_gap)[0]
        self._orig._tokens_offsets = Tokenizer._remove_gap_offsets([o_off], self._has_bos_token, keep_final_gap)[0]
        self._edit._tokens_offsets = Tokenizer._remove_gap_offsets([e_off], self._has_bos_token, keep_final_gap)[0]
        self._has_gaps = False
        self._has_eos_token = keep_final_gap

    ### Helper Functions ###

    def _get_edit_str(self):
        orig_tokens_str = str(self._orig.labeled_tokens).replace("\n", "\n" + 8 * " ")
        edit_tokens_str = str(self._edit.labeled_tokens).replace("\n", "\n" + 8 * " ")
        aligned_str = self.aligned_str.replace("\n", "\n" + 8 * " ")
        out_str = f"""\
        edit.text:
        {indent(self._edit._text, 12 * " ")}
        orig.tokens:
        {indent(orig_tokens_str, 12 * " ")}
        edit.tokens:
        {indent(edit_tokens_str, 12 * " ")}
        aligned:
        {indent(aligned_str, 12 * " ")}
        """
        return out_str.strip()

    @classmethod
    def _get_editing_stats(cls, aligned: WordOutput, use_rich: bool = False) -> str:
        """Return the editing statistics based on the alignment type using Jiwer."""
        unit_rate_link = "https://docs.kolena.com/metrics/wer-cer-mer/#word-error-rate"
        metrics_str = ""
        metrics_str += "=== Categories ==="
        metrics_str += f"\nCorrect: {aligned.hits} token(s)"
        metrics_str += f"\nSubstitutions (S): {aligned.substitutions} token(s)"
        metrics_str += f"\nInsertions (I): {aligned.insertions} token(s)"
        metrics_str += f"\nDeletions (D): {aligned.deletions} token(s)"
        metrics_str += "\n\n=== Metrics ==="
        if use_rich:
            metrics_str += f"\n[link={unit_rate_link}]Word Error Rate (WER)[/link]: {aligned.wer}"
            metrics_str += f"\n[link=https://docs.kolena.com/metrics/wer-cer-mer/#match-error-rate]Match Error Rate (MER)[/link]: {aligned.mer}"
            metrics_str += f"\n[link=https://lightning.ai/docs/torchmetrics/stable/text/word_info_lost.html]Word Information Lost (WIL)[/link]: {aligned.wil}"
            metrics_str += f"\n[link=https://lightning.ai/docs/torchmetrics/stable/text/word_info_preserved.html]Word Information Preserved (WIP)[/link]: {aligned.wip}"
        return metrics_str

    @staticmethod
    def _get_tokens_with_gaps(
        t: Tokenizer, text: str, toks: list[str] | None, offs: list[tuple[int, int] | None] | None, gap_token: str
    ):
        if toks is None or offs is None:
            all_toks, all_offs = t.tokenize_with_offsets(text, add_gaps=True, gap_token=gap_token)
        elif not t._has_gaps(toks, offs):
            all_toks, all_offs = t._add_gaps_to_tokens_and_offsets([toks], [offs], gap_token=gap_token)
        else:
            all_toks, all_offs = [toks], [offs]
        return all_toks[0], all_offs[0]

    def _get_label_types(self) -> list[type]:
        return list(
            {type(l) for l in list(self._orig._tokens_labels) + list(self._edit._tokens_labels) if l is not None}
        )

    def _relabel_attributes(
        self,
        relabel_fn: Callable[[LabelType], LabelType],
    ) -> None:
        self._orig._relabel_attributes(relabel_fn=relabel_fn)
        self._edit._relabel_attributes(relabel_fn=relabel_fn)

    def _get_labels_array(
        self,
        items: "Sequence[EditedEntry]",
        dtype: type | None = None,
    ) -> npt.NDArray[np.str_ | np.integer | np.floating]:
        all_labels = []
        for item in items:
            item_labels = []
            num_tokens = len(item.orig.tokens_labels)
            for idx, label in enumerate(item.orig.tokens_labels):
                if (idx == 0 and item.has_bos_token) or (idx == num_tokens - 1 and item.has_eos_token):
                    # BOS/EOS tokens are not labeled
                    item_labels.append(None)
                else:
                    item_labels.append(label if label is not None else np.nan)
            all_labels.append(item_labels)
        labels_array = np.array(all_labels)
        return labels_array.astype(dtype)


class MultiEditEntry(BaseMultiLabelEntry[EditedEntry]):
    """Class for a list of `EditedEntry` representing multiple edits over the same `orig` text."""

    def __str__(self) -> str:
        if len(self) == 0:
            return "No entries available."
        out_str = dedent(f"""\
          orig.text:
        {indent(self[0].orig.text, 12 * " ")}
        """)
        for i, entry in enumerate(self):
            out_str += dedent(f"""\
        === Edit #{i} ===
        {entry._get_edit_str()}
        """)
        return out_str.strip()

    ### Utility Methods ###

    def merge_gap_annotations(
        self,
        merge_fn: Callable[[Sequence[LabelType]], LabelType] | None = None,
        keep_final_gap: bool = True,
    ) -> None:
        """Merge gap annotations in the tokens of `orig` and `edit`.

        This method is equivalent to calling `EditedEntry.from_edits` with `with_gaps=False`. Gap annotations are merged
        to the next non-gap token to the right, and the gap label is added to the label of the non-gap token. The last
        gap is kept to account for insertions at the end of the text.

        E.g. `GAP Hello GAP World GAP ! GAP` becomes `Hello World ! GAP`.
             `  I     S   I               I`         `   IS     I     I`
        """
        for entry in self:
            entry.merge_gap_annotations(merge_fn=merge_fn, keep_final_gap=keep_final_gap)
