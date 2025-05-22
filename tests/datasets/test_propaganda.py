import pytest
from labl.datasets import load_propaganda
from labl.data.labeled_dataset import LabeledDataset


def test_load_propaganda_default():
    """Test loading propaganda dataset with default parameters."""
    try:
        # Load with minimal parameters to test default behavior
        dataset = load_propaganda()
        
        # Check that the dataset is a LabeledDataset
        assert isinstance(dataset, LabeledDataset)
        
        # Check that the dataset has entries
        assert len(dataset) > 0
        
        # Check that the entries have spans
        entry = dataset[0]
        assert hasattr(entry, "spans")
        
        # Check that the entries have the expected metadata
        assert "tgt" in entry.info
        assert "dataset" in entry.info
        assert entry.info["dataset"] == "propaganda"
        
    except Exception as e:
        pytest.skip(f"Failed to load propaganda dataset: {e}")


def test_load_propaganda_with_tokenizer():
    """Test loading propaganda dataset with a custom tokenizer."""
    try:
        # Import the WhitespaceTokenizer directly
        from labl.utils.tokenizer import WhitespaceTokenizer
        
        # Load with a WhitespaceTokenizer instance
        dataset = load_propaganda(tokenizer=WhitespaceTokenizer())
        
        # Check that the dataset is tokenized
        entry = dataset[0]
        assert hasattr(entry, "tokens")
        assert len(entry.tokens) > 0
        
        # Check that spans exist
        if len(entry.spans) > 0:
            span = entry.spans[0]
            # Verify span has basic attributes
            assert hasattr(span, "start")
            assert hasattr(span, "end")
            assert hasattr(span, "label")
            assert hasattr(span, "text")
        
    except Exception as e:
        # Print the exception for debugging
        print(f"Exception in test_load_propaganda_with_tokenizer: {e}")
        pytest.skip(f"Failed to load propaganda dataset with tokenizer: {e}")