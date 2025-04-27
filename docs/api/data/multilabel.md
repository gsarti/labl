::: wqe.data.base_sequence.BaseMultiLabelEntry
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - label_counts
        - tokens_with_label_counts
        - get_agreement


::: wqe.data.labeled_entry.MultiLabelEntry
    handler: python
    options:
      show_root_heading: true
      show_source: true


::: wqe.data.edited_entry.MultiEditEntry
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - merge_gap_annotations
