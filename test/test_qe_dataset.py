from wqeval import WordLevelQEDataset, WordLevelQEEntry


def test_wqe_entry():
    entry = WordLevelQEEntry(
        src="The quick brown fox jumps over the lazy dog.",
        mt="La zorra bruna lesta saltò sobre el perro perezoso.",
        pe="La zorra marrón rápida salta sobre el perro perezosa.",
    )
    assert isinstance(entry, WordLevelQEEntry)


def test_wqe_dataset():
    dataset = WordLevelQEDataset.from_sentences(
        src=["Hello world!", "Goodbye world!", "Awesome, this works"],
        mt=["Holee planeta!", "Adiós carajo!", "Genial"],
        pe=["Hola mundo!", "Adiós mundo!", "Genial, esto funciona"],
    )
    assert isinstance(dataset, WordLevelQEDataset)
    assert len(dataset) == 3
    assert isinstance(dataset[0], WordLevelQEEntry)
    assert isinstance(dataset.word_stats, str)
