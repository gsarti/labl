::: wqe.data.tokenizer.Tokenizer
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - __call__
      - tokenize
      - detokenize
      - tokenize_with_offsets

::: wqe.data.tokenizer.WhitespaceTokenizer
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - detokenize
      - tokenize_with_offsets

::: wqe.data.tokenizer.WordBoundaryTokenizer
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - detokenize
      - tokenize_with_offsets

::: wqe.data.tokenizer.HuggingfaceTokenizer
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
      - detokenize
      - tokenize_with_offsets
