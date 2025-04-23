from wqe.utils.span import Span


def test_span_init():
    span = Span(start=0, end=5, label="OK")
    assert span.start == 0
    assert span.end == 5
    assert span.label == "OK"
    assert span.text is None
    span_with_text = Span(start=0, end=5, label="OK", text="Hello")
    assert span_with_text.text == "Hello"


def test_span_to_dict():
    span = Span(start=0, end=5, label="OK", text="Hello")
    span_dict = span.to_dict()
    assert span_dict["start"] == 0
    assert span_dict["end"] == 5
    assert span_dict["label"] == "OK"
    assert span_dict["text"] == "Hello"


def test_span_load():
    span_dict = {"start": 0, "end": 5, "label": "OK", "text": "Hello"}
    span = Span.from_dict(span_dict)
    assert span.start == 0
    assert span.end == 5
    assert span.label == "OK"
    assert span.text == "Hello"


def test_span_from_list():
    span_dicts = [
        {"start": 0, "end": 5, "label": "OK", "text": "Hello"},
        {"start": 6, "end": 11, "label": "BAD", "text": "World"},
    ]
    spans = Span.from_list(span_dicts)
    assert len(spans) == 2
    assert spans[0].start == 0
    assert spans[0].label == "OK"
    assert spans[1].end == 11
    assert spans[1].label == "BAD"
