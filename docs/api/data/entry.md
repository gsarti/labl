::: labl.data.base_entry.BaseLabeledEntry
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - relabel
        - get_agreement
        - get_correlation

::: labl.data.labeled_entry.LabeledEntry
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - text
        - spans
        - tagged
        - tokens
        - tokens_labels
        - tokens_offsets
        - labeled_tokens
        - from_spans
        - from_tagged
        - from_tokens
        - get_tagged_from_spans
        - get_tokens_from_spans
        - get_text_and_spans_from_tagged
        - get_spans_from_tokens
        - get_tokens
        - get_labels
        - to_dict
        - from_dict

::: labl.data.edited_entry.EditedEntry
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - orig
        - edit
        - aligned
        - has_gaps
        - has_bos_token
        - has_eos_token
        - aligned_str
        - from_edits
        - get_tokens_labels_from_edit
        - get_tokens
        - get_labels
        - to_dict
        - from_dict
        - merge_gap_annotations
