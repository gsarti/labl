from wqe.utils.transform import ReduceToListOfListOfTokens, RegexReduceToListOfListOfWords


def test_regex_reduce_init():
    # Test default initialization
    transform = RegexReduceToListOfListOfWords()
    assert transform.exp is not None

    # Test custom pattern
    custom_pattern = r"[,.!?;:]|\s+"
    transform = RegexReduceToListOfListOfWords(exp=custom_pattern)
    assert transform.exp == custom_pattern


def test_regex_reduce_process_string():
    transform = RegexReduceToListOfListOfWords()

    # Test with simple sentence
    text = "Hello, world!"
    result = transform.process_string(text)
    assert result == [["Hello", ",", "world", "!"]]
    text = ""
    result = transform.process_string(text)
    assert result == [[]]


def test_token_reduce_init():
    # Test default initialization
    transform = ReduceToListOfListOfTokens("bert-base-uncased")
    assert transform.tokenizer is not None

    # Test custom tokenizer
    custom_tokenizer = "distilbert-base-uncased"
    transform = ReduceToListOfListOfTokens(tokenizer_or_id=custom_tokenizer)
    assert transform.tokenizer.__class__.__name__ == "DistilBertTokenizerFast"


def test_token_reduce_process_string():
    transform = ReduceToListOfListOfTokens("bert-base-uncased")
    text = "Hello, world!"
    result = transform.process_string(text)
    assert result == [["hello", ",", "world", "!"]]


def test_token_reduce_process_string_with_special_tokens():
    transform = ReduceToListOfListOfTokens("bert-base-uncased", add_special_tokens=True)
    text = "Hello, world!"
    result = transform.process_string(text)
    assert result == [["[CLS]", "hello", ",", "world", "!", "[SEP]"]]
