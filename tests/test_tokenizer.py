from pytest import fixture

from wqe.utils.tokenizer import HuggingfaceTokenizer, WhitespaceTokenizer, WordBoundaryTokenizer


@fixture
def nllb_tokenizer() -> HuggingfaceTokenizer:
    return HuggingfaceTokenizer(
        "facebook/nllb-200-3.3B", src_lang="eng_Latn", tgt_lang="ita_Latn", add_special_tokens=True
    )


def test_whitespace_tokenizer_init():
    tokenizer = WhitespaceTokenizer()
    txt = "Hello world"
    tokens = tokenizer(txt)
    assert tokens == [["Hello", "world"]]
    custom_tokenizer = WhitespaceTokenizer(word_delimiter="|")
    txt = "Hello|world"
    tokens = custom_tokenizer(txt)
    assert tokens == [["Hello", "world"]]


def test_whitespace_tokenizer_tokenize():
    tokenizer = WhitespaceTokenizer()
    text = "Hello world"
    tokens = tokenizer.tokenize(text)
    assert tokens == [["Hello", "world"]]

    # Test with list input
    tokens = tokenizer.tokenize(["Hello world", "Another sentence"])
    assert tokens == [["Hello", "world"], ["Another", "sentence"]]


def test_whitespace_tokenizer_detokenize():
    tokenizer = WhitespaceTokenizer()
    tokens = ["Hello", "world"]
    text = tokenizer.detokenize(tokens)
    assert text == ["Hello world"]
    tokens = [["Hello", "world"], ["Another", "sentence"]]
    text = tokenizer.detokenize(tokens)
    assert text == ["Hello world", "Another sentence"]


def test_whitespace_tokenizer_tokenize_with_offsets():
    tokenizer = WhitespaceTokenizer()
    text = "Hello world"
    tokens, offsets = tokenizer.tokenize_with_offsets(text)
    assert tokens == [["Hello", "world"]]
    assert offsets == [[(0, 5), (6, 11)]]
    tokens, offsets = tokenizer.tokenize_with_offsets(["Hello world", "Another sentence"])
    assert tokens == [["Hello", "world"], ["Another", "sentence"]]
    assert offsets == [[(0, 5), (6, 11)], [(0, 7), (8, 16)]]


def test_word_boundary_tokenizer_tokenize():
    tokenizer = WordBoundaryTokenizer()
    text = "Hello, world!"
    tokens = tokenizer.tokenize(text)
    assert tokens == [["Hello", ",", "world", "!"]]
    tokens = tokenizer.tokenize(["Hello, world!", "Another sentence."])
    assert tokens == [["Hello", ",", "world", "!"], ["Another", "sentence", "."]]


def test_word_boundary_tokenizer_detokenize():
    tokenizer = WordBoundaryTokenizer()
    tokens = ["Hello", ",", "world", "!"]
    text = tokenizer.detokenize(tokens)
    assert text == ["Hello, world!"]

    # Test with nested list
    tokens = [["Hello", ",", "world", "!"], ["Another", "sentence", "."]]
    text = tokenizer.detokenize(tokens)
    assert text == ["Hello, world!", "Another sentence."]


def test_word_boundary_tokenizer_tokenize_with_offsets():
    tokenizer = WordBoundaryTokenizer()
    text = "Hello, world!"
    tokens, offsets = tokenizer.tokenize_with_offsets(text)
    assert tokens == [["Hello", ",", "world", "!"]]
    assert offsets == [[(0, 5), (5, 6), (7, 12), (12, 13)]]
    tokens, offsets = tokenizer.tokenize_with_offsets(["Hello, world!", "Another sentence."])
    assert tokens == [["Hello", ",", "world", "!"], ["Another", "sentence", "."]]
    assert offsets == [[(0, 5), (5, 6), (7, 12), (12, 13)], [(0, 7), (8, 16), (16, 17)]]


def test_word_boundary_tokenizer_call():
    tokenizer = WordBoundaryTokenizer()
    text = "Hello, world!"
    tokens = tokenizer(text)
    assert tokens == [["Hello", ",", "world", "!"]]
    tokens, offsets = tokenizer(text, with_offsets=True)
    assert tokens == [["Hello", ",", "world", "!"]]
    assert offsets == [[(0, 5), (5, 6), (7, 12), (12, 13)]]
    # Test with list input
    tokens = tokenizer(["Hello, world!", "Another sentence."])
    assert tokens == [["Hello", ",", "world", "!"], ["Another", "sentence", "."]]
    tokens, offsets = tokenizer(["Hello, world!", "Another sentence."], with_offsets=True)
    assert tokens == [["Hello", ",", "world", "!"], ["Another", "sentence", "."]]
    assert offsets == [[(0, 5), (5, 6), (7, 12), (12, 13)], [(0, 7), (8, 16), (16, 17)]]


def test_huggingface_tokenizer_init():
    tokenizer = HuggingfaceTokenizer("bert-base-uncased")
    text = "Hello world"
    tokens = tokenizer(text)
    assert tokens == [["hello", "world"]]


def test_huggingface_tokenizer_init_with_kwargs(nllb_tokenizer: HuggingfaceTokenizer):
    text = "Buongiorno mondo!"
    tokens = nllb_tokenizer(text)
    assert tokens == [["ita_Latn", "▁Bu", "ongi", "orno", "▁mondo", "!", "</s>"]]


def test_huggingface_tokenizer_tokenize(nllb_tokenizer: HuggingfaceTokenizer):
    text = "Buongiorno mondo!"
    tokens = nllb_tokenizer.tokenize(text)
    assert tokens == [["ita_Latn", "▁Bu", "ongi", "orno", "▁mondo", "!", "</s>"]]
    tokens = nllb_tokenizer.tokenize(["Hello world", "Another sentence"])
    assert tokens == [
        ["ita_Latn", "▁Hello", "▁world", "</s>"],
        ["ita_Latn", "▁Another", "▁sentence", "</s>"],
    ]


def test_huggingface_tokenizer_detokenize(nllb_tokenizer: HuggingfaceTokenizer):
    tokens = ["ita_Latn", "▁Bu", "ongi", "orno", "▁mondo", "!"]
    text = nllb_tokenizer.detokenize(tokens)
    assert text == ["Buongiorno mondo!"]
    tokens = [["ita_Latn", "▁Bu", "ongi", "orno", "▁mondo", "!"], ["ita_Latn", "</s>"]]
    text = nllb_tokenizer.detokenize(tokens)
    assert text == ["Buongiorno mondo!", ""]


def test_huggingface_tokenizer_tokenize_with_offsets(nllb_tokenizer: HuggingfaceTokenizer):
    text = "Buongiorno mondo!"
    tokens, offsets = nllb_tokenizer.tokenize_with_offsets(text)
    assert tokens == [["ita_Latn", "▁Bu", "ongi", "orno", "▁mondo", "!", "</s>"]]
    assert offsets == [[None, (0, 2), (2, 6), (6, 10), (10, 16), (16, 17), None]]
    tokens, offsets = nllb_tokenizer.tokenize_with_offsets(["Hello world", "Another sentence"])
    assert tokens == [
        ["ita_Latn", "▁Hello", "▁world", "</s>"],
        ["ita_Latn", "▁Another", "▁sentence", "</s>"],
    ]
    assert offsets == [[None, (0, 5), (5, 11), None], [None, (0, 7), (7, 16), None]]
