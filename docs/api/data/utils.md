# Data Utils

## 🔀 Mixins

::: wqe.data.mixin.AlignedSequencesMixin
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - aligned
      - aligned_char
      - aligned_str
      - aligned_char_str

## 🔤 Spans

::: wqe.data.span.BaseSpan
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - load
      - from_list
      - to_dict

::: wqe.data.span.Span
    handler: python
    options:
      show_root_heading: true
      show_source: false

::: wqe.data.span.EditSpan
    handler: python
    options:
      show_root_heading: true
      show_source: false

## 🔠 Tokens

::: wqe.data.token.LabeledToken
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - to_tuple
      - from_tuple
      - from_list

## 🔄 Transforms

::: wqe.data.transform.RegexReduceToListOfListOfWords
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: wqe.data.transform.ReduceToListOfListOfTokens
    handler: python
    options:
      show_root_heading: true
      show_source: true

## 🤝 Aggregation Functions

::: wqe.data.aggregation.LabelAggregation
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - __call__

::: wqe.data.aggregation.label_count_aggregation
    handler: python
    options:
      show_root_heading: true
      show_source: true
