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
      show_source: true
      members:
      - load
      - from_list
      - to_dict

::: wqe.data.span.Span
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: wqe.data.span.EditSpan
    handler: python
    options:
      show_root_heading: true
      show_source: true

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

::: wqe.data.token.LabeledTokenList
    handler: python
    options:
      show_root_heading: true
      show_source: false

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
