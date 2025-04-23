from wqe.utils.token import LabeledToken, LabeledTokenList


def test_labeled_token_init():
    token = LabeledToken(token="hello", label="OK")
    assert token.token == "hello"
    assert token.label == "OK"
    assert token.t == "hello"  # shortcut property
    assert token.l == "OK"  # shortcut property


def test_labeled_token_str():
    token = LabeledToken(token="hello", label="OK")
    assert str(token) == "(hello, OK)"
    token = LabeledToken(token="hello", label=None)
    assert str(token) == "(hello, None)"


def test_labeled_token_to_tuple():
    token = LabeledToken(token="hello", label="OK")
    assert token.to_tuple() == ("hello", "OK")


def test_labeled_token_from_tuple():
    token = LabeledToken.from_tuple(("hello", "OK"))
    assert token.token == "hello"
    assert token.label == "OK"


def test_labeled_token_from_list_tuples():
    tokens = LabeledToken.from_list([("hello", "OK"), ("world", "BAD")])
    assert len(tokens) == 2
    assert tokens[0].token == "hello"
    assert tokens[0].label == "OK"
    assert tokens[1].token == "world"
    assert tokens[1].label == "BAD"


def test_labeled_token_from_list_strings():
    tokens = LabeledToken.from_list([("hello", None), ("world", "BAD")])
    assert len(tokens) == 2
    assert tokens[0].token == "hello"
    assert tokens[0].label is None
    assert tokens[1].token == "world"
    assert tokens[1].label == "BAD"


def test_labeled_token_from_list_with_ignore_labels():
    tokens = LabeledToken.from_list(
        [("hello", "OK"), ("world", "BAD"), ("test", "NEUTRAL")], ignore_labels=["NEUTRAL"]
    )
    assert len(tokens) == 3
    assert tokens[0].label == "OK"
    assert tokens[1].label == "BAD"
    assert tokens[2].label is None


def test_labeled_token_list_init():
    token_list = LabeledTokenList([LabeledToken("hello", "OK"), LabeledToken("world", "BAD")])
    assert len(token_list) == 2
    assert token_list[0].token == "hello"
    assert token_list[1].label == "BAD"


def test_labeled_token_list_str():
    token_list = LabeledTokenList([LabeledToken("hello", "OK"), LabeledToken("world", "BAD")])
    expected_str = "hello world\n   OK   BAD\n"
    assert str(token_list) == expected_str

    # Test with None labels
    token_list = LabeledTokenList([LabeledToken("hello", None), LabeledToken("world", "BAD")])
    expected_str = "hello world\n        BAD\n"
    assert str(token_list) == expected_str
