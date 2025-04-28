# Utility Classes

## 🔤 Spans

::: labl.utils.span.Span
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - from_dict
      - from_list
      - to_dict

::: labl.utils.span.SpanList
    handler: python
    options:
      show_root_heading: true
      show_source: false

## 🔠 Tokens

::: labl.utils.token.LabeledToken
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - from_tuple
      - from_list
      - to_tuple

::: labl.utils.token.LabeledTokenList
    handler: python
    options:
      show_root_heading: true
      show_source: false

## 🔄 Transforms

::: labl.utils.transform.RegexReduceToListOfListOfWords
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: labl.utils.transform.ReduceToListOfListOfTokens
    handler: python
    options:
      show_root_heading: true
      show_source: true

## 🤝 Aggregation Functions

::: labl.utils.aggregation.LabelAggregation
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - __call__

::: labl.utils.aggregation.label_sum_aggregation
    handler: python
    options:
      show_root_heading: true
      show_source: true

## 🏷️ LabeledInterface

::: labl.data.labeled_interface.LabeledInterface
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
      - label_types
      - relabel
