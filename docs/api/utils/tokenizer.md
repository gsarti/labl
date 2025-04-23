::: wqe.utils.tokenizer.Tokenizer
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - __call__
      - tokenize
      - detokenize
      - tokenize_with_offsets

::: wqe.utils.tokenizer.WhitespaceTokenizer
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - detokenize
      - tokenize_with_offsets

::: wqe.utils.tokenizer.WordBoundaryTokenizer
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - detokenize
      - tokenize_with_offsets

::: wqe.utils.tokenizer.HuggingfaceTokenizer
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - detokenize
      - tokenize_with_offsets
