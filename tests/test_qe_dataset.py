from wqeval import WordLevelQEDataset, WordLevelQEEntry


def test_wqe_entry():
    entry = WordLevelQEEntry(
        orig="La zorra bruna lesta saltò sobre el perro perezoso.",
        edit="La zorra marrón rápida salta sobre el perro perezosa.",
    )
    assert isinstance(entry, WordLevelQEEntry)


def test_wqe_dataset():
    dataset = WordLevelQEDataset.from_sentences(
        orig=["Holee planeta!", "Adiós carajo!", "Genial"],
        edit=["Hola mundo!", "Adiós mundo!", "Genial, esto funciona"],
    )
    assert isinstance(dataset, WordLevelQEDataset)
    assert len(dataset) == 3
    assert isinstance(dataset[0], WordLevelQEEntry)
    assert isinstance(dataset.word_stats, str)
