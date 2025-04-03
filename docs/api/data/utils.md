# Data Utils

## 🔀 Mixins

::: wqe.data.aligned_mixin.AlignedSequencesMixin
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

::: wqe.data.qe_span.AbstractSpan
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
      - load
      - from_list
      - to_dict

::: wqe.data.qe_span.QESpan
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: wqe.data.qe_span.QESpanWithEdit
    handler: python
    options:
      show_root_heading: true
      show_source: true

## 🔠 Tokens

::: wqe.data.tokenizer.LabeledToken
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - to_tuple
      - from_tuple
      - from_list

## 🔄 Transforms

::: wqe.data.tokenizer.RegexReduceToListOfListOfWords
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: wqe.data.tokenizer.ReduceToListOfListOfTokens
    handler: python
    options:
      show_root_heading: true
      show_source: true
