# Dataset

::: labl.data.base_sequence.BaseLabeledDataset
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - get_agreement

::: labl.data.labeled_dataset.LabeledDataset
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - from_spans
        - from_tagged
        - from_tokens
        - get_label_agreement

::: labl.data.edited_dataset.EditedDataset
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - from_edits
        - from_edits_dataframe
        - merge_gap_annotations
