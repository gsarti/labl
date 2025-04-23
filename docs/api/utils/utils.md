# Utility Classes

## 🔤 Spans

::: wqe.utils.span.Span
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - from_dict
      - from_list
      - to_dict

::: wqe.utils.span.SpanList
    handler: python
    options:
      show_root_heading: true
      show_source: false

## 🔠 Tokens

::: wqe.utils.token.LabeledToken
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - from_tuple
      - from_list
      - to_tuple

::: wqe.utils.token.LabeledTokenList
    handler: python
    options:
      show_root_heading: true
      show_source: false

## 🔄 Transforms

::: wqe.utils.transform.RegexReduceToListOfListOfWords
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: wqe.utils.transform.ReduceToListOfListOfTokens
    handler: python
    options:
      show_root_heading: true
      show_source: true

## 🤝 Aggregation Functions

::: wqe.utils.aggregation.LabelAggregation
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - __call__

::: wqe.utils.aggregation.label_sum_aggregation
    handler: python
    options:
      show_root_heading: true
      show_source: true
