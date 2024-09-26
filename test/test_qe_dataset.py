from intrinsic_qe import MTPEDataset, MTPEEntry

ALIGNED_ENTRY_STRING_EXAMPLE = """SRC: The quick brown fox jumps over the lazy dog.

 MT: La zorra  bruna  lesta saltò sobre el perro perezoso.
 PE: La zorra marrón rápida salta sobre el perro perezosa.
                   S      S     S                        S

=== Categories ===
Correct: 5 words
Substitutions (S): 4 words
Insertions (I): 0 words
Deletions (D): 0 words

=== Metrics ===
Word Error Rate (WER): 0.4444444444444444
Match Error Rate (MER): 0.4444444444444444
Word Information Lost (WIL): 0.691358024691358
Word Information Preserved (WIP): 0.308641975308642"""

ALIGNED_DATASET_STRING_EXAMPLE = """#1
SRC: Hello world!

 MT: Holee planeta!
 PE:  Hola   mundo!
         S        S

#2
SRC: Goodbye world!

 MT: Adiós carajo!
 PE: Adiós  mundo!
                 S

#3
SRC: Awesome, this works

 MT: ******* ****   Genial
 PE: Genial, esto funciona
           I    I        S

=== Categories ===
Correct: 1 word
Substitutions (S): 4 words
Insertions (I): 2 words
Deletions (D): 0 words

=== Metrics ===
Word Error Rate (WER): 1.2
Match Error Rate (MER): 0.8571428571428571
Word Information Lost (WIL): 0.9714285714285714
Word Information Preserved (WIP): 0.02857142857142857"""

ENTRY_CHAR_STATS = """=== Categories ===
Correct: 6 characters
Substitutions (S): 5 characters
Insertions (I): 0 characters
Deletions (D): 3 characters

=== Metrics ===
Character Error Rate (CER): 0.5714285714285714"""

DATASET_WORD_STATS = """=== Categories ===
Correct: 1 word
Substitutions (S): 4 words
Insertions (I): 2 words
Deletions (D): 0 words

=== Metrics ===
Word Error Rate (WER): 1.2
Match Error Rate (MER): 0.8571428571428571
Word Information Lost (WIL): 0.9714285714285714
Word Information Preserved (WIP): 0.02857142857142857"""


def test_mtpe_entry():
    entry = MTPEEntry(
        source="The quick brown fox jumps over the lazy dog.",
        mt="La zorra bruna lesta saltò sobre el perro perezoso.",
        pe="La zorra marrón rápida salta sobre el perro perezosa.",
    )
    assert str(entry) == ALIGNED_ENTRY_STRING_EXAMPLE


def test_mtpe_dataset():
    dataset = MTPEDataset.from_sentences(
        source=["Hello world!", "Goodbye world!", "Awesome, this works"],
        mt=["Holee planeta!", "Adiós carajo!", "Genial"],
        pe=["Hola mundo!", "Adiós mundo!", "Genial, esto funciona"],
    )
    assert str(dataset) == ALIGNED_DATASET_STRING_EXAMPLE
    assert dataset[0].char_stats == ENTRY_CHAR_STATS
    assert dataset.word_stats == DATASET_WORD_STATS
