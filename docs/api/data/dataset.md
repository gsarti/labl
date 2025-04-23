# Dataset

::: wqe.data.base_dataset.BaseDataset
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - label_types
        - relabel

::: wqe.data.labeled_dataset.LabeledDataset
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - from_spans
        - from_tagged
        - from_tokens
        - get_label_agreement

::: wqe.data.edited_dataset.EditedDataset
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - from_edits
        - from_edits_dataframe
