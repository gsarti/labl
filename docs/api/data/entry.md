# Entry

::: wqe.data.base_entry.BaseEntry
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - relabel

::: wqe.data.labeled_entry.LabeledEntry
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - labeled_tokens
        - from_spans
        - from_tagged
        - from_tokens
        - get_tagged_from_spans
        - get_tokens_from_spans
        - get_text_and_spans_from_tagged
        - get_spans_from_tokens

::: wqe.data.edited_entry.EditedEntry
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - aligned_str
        - from_edits
        - get_tokens_labels_from_edit
        - merge_gap_annotations

::: wqe.data.edited_entry.MultiEditedEntry
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - merge_gap_annotations
        - edits_counts
        - tokens_with_edits_counts
