"""Extends jiwer to support token pre-computation"""

from typing import cast

from jiwer import AbstractTransform, AlignmentChunk, Compose, WordOutput
from jiwer.process import _apply_transform, _word2int
from jiwer.transformations import wer_default
from rapidfuzz.distance.Levenshtein import opcodes as get_opcodes


def process_words(
    texts: str | list[str] | list[list[str]],
    edits: str | list[str] | list[list[str]],
    texts_transform: Compose | AbstractTransform = wer_default,
    edits_transform: Compose | AbstractTransform = wer_default,
    is_text_pre_transformed: bool = False,
    is_edit_pre_transformed: bool = False,
) -> WordOutput:
    """
    Compute the word-level levenshtein distance and alignment between one or more
    texts and edits. Based on the result, multiple measures can be computed, such as the word error rate.

    Args:
        texts (str | list[str]): The original text(s)
        edits (str | list[str]): The edited text(s)
        texts_transform (Compose | AbstractTransform): The transformation(s) to apply to the texts string(s)
        edits_transform (Compose | AbstractTransform): The transformation(s) to apply to the edits string(s)
        is_text_pre_transformed (bool): If True, the text is already pre-transformed.
        is_edit_pre_transformed (bool): If True, the edit is already pre-transformed.

    Returns:
        The processed texts and edits.

    Raises:
        ValueError: If one or more texts are empty strings
        ValueError: If after applying transforms, texts and edits lengths don't match
    """
    # validate input type
    if isinstance(texts, str):
        texts = [texts]
    if isinstance(edits, str):
        edits = [edits]
    if any(len(t) == 0 for t in texts):
        raise ValueError("one or more text are empty strings")

    # pre-process reference and hypothesis by applying transforms
    if not is_text_pre_transformed:
        texts = cast(list[str], texts)
        text_transformed = _apply_transform(texts, texts_transform, is_reference=True)
    else:
        text_transformed = texts
    if not is_edit_pre_transformed:
        edits = cast(list[str], edits)
        edit_transformed = _apply_transform(edits, edits_transform, is_reference=False)
    else:
        edit_transformed = edits
    text_transformed = cast(list[list[str]], text_transformed)
    edit_transformed = cast(list[list[str]], edit_transformed)

    if len(text_transformed) != len(edit_transformed):
        raise ValueError(
            "After applying the transforms on the text and edit sentences, "
            f"their lengths must match. "
            f"Instead got {len(text_transformed)} texts and "
            f"{len(edit_transformed)} edits."
        )

    # Map each word into a unique integer in order to compute
    # word-level levenshtein distance
    ref_as_ints, hyp_as_ints = _word2int(text_transformed, edit_transformed)

    # keep track of total hits, substitutions, deletions and insertions
    # across all input sentences
    num_hits, num_substitutions, num_deletions, num_insertions = 0, 0, 0, 0

    # also keep track of the total number of words in the reference and hypothesis
    num_rf_words, num_hp_words = 0, 0

    # anf finally, keep track of the alignment between each reference and hypothesis
    alignments = []

    for reference_sentence, hypothesis_sentence in zip(ref_as_ints, hyp_as_ints, strict=False):
        # Get the opcodes directly
        opcodes = get_opcodes(reference_sentence, hypothesis_sentence)

        subs = dels = ins = hits = 0
        sentence_op_chunks = []

        for tag, i1, i2, j1, j2 in opcodes:
            tag = cast(str, tag)
            i1 = cast(int, i1)
            i2 = cast(int, i2)
            j1 = cast(int, j1)
            j2 = cast(int, j2)
            # Create alignment chunk
            sentence_op_chunks.append(
                AlignmentChunk(
                    type=tag,
                    ref_start_idx=i1,
                    ref_end_idx=i2,
                    hyp_start_idx=j1,
                    hyp_end_idx=j2,
                )
            )

            # Update counts
            if tag == "equal":
                hits += i2 - i1
            elif tag == "replace":
                subs += i2 - i1
            elif tag == "delete":
                dels += i2 - i1
            elif tag == "insert":
                ins += j2 - j1

        # Update global counts
        num_hits += hits
        num_substitutions += subs
        num_deletions += dels
        num_insertions += ins
        num_rf_words += len(reference_sentence)
        num_hp_words += len(hypothesis_sentence)
        alignments.append(sentence_op_chunks)

    # Compute all measures
    S, D, I, H = num_substitutions, num_deletions, num_insertions, num_hits

    wer = float(S + D + I) / float(H + S + D)
    mer = float(S + D + I) / float(H + S + D + I)
    wip = (float(H) / num_rf_words) * (float(H) / num_hp_words) if num_hp_words >= 1 else 0
    wil = 1 - wip

    # return all output
    return WordOutput(
        references=text_transformed,
        hypotheses=edit_transformed,
        alignments=alignments,
        wer=wer,
        mer=mer,
        wil=wil,
        wip=wip,
        hits=num_hits,
        substitutions=num_substitutions,
        insertions=num_insertions,
        deletions=num_deletions,
    )
