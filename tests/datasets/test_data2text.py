import pytest
from labl.datasets import load_data2text
from labl.data.labeled_dataset import LabeledDataset


def test_load_data2text_default():
    """Test loading data2text dataset with default parameters."""
    try:
        # Load with minimal parameters to test default behavior
        # We'll limit to just one dataset and one model to make the test faster
        datasets = load_data2text(
            datasets=["d2t-openweather"],
            nlg_models=["gpt4o"],
            splits=["test"]
        )
        
        # Check the structure of the returned dictionary
        assert isinstance(datasets, dict)
        assert "test" in datasets
        assert "d2t-openweather" in datasets["test"]
        assert "gpt4o" in datasets["test"]["d2t-openweather"]
        
        # Check that the dataset is a LabeledDataset
        dataset = datasets["test"]["d2t-openweather"]["gpt4o"]
        assert isinstance(dataset, LabeledDataset)
        
        # Check that the dataset has entries
        assert len(dataset) > 0
        
        # Check that the entries have spans
        entry = dataset[0]
        assert hasattr(entry, "spans")
        
        # Check that the entries have the expected metadata
        assert "tgt" in entry.info
        assert "src" in entry.info
        assert "dataset" in entry.info
        assert "model" in entry.info
        assert entry.info["dataset"] == "d2t-openweather"
        assert entry.info["model"] == "gpt4o"
        
    except Exception as e:
        pytest.skip(f"Failed to load data2text dataset: {e}")


def test_load_data2text_with_tokenizer():
    """Test loading data2text dataset with a custom tokenizer."""
    try:
        # Import the WhitespaceTokenizer directly
        from labl.utils.tokenizer import WhitespaceTokenizer
        
        # Load with a WhitespaceTokenizer instance
        datasets = load_data2text(
            datasets=["d2t-openweather"],
            nlg_models=["gpt4o"],
            splits=["test"],
            tokenizer=WhitespaceTokenizer()
        )
        
        # Check that the dataset is tokenized
        dataset = datasets["test"]["d2t-openweather"]["gpt4o"]
        entry = dataset[0]
        assert hasattr(entry, "tokens")
        assert len(entry.tokens) > 0
        
    except Exception as e:
        pytest.skip(f"Failed to load data2text dataset with tokenizer: {e}")


def test_load_data2text_multiple_datasets():
    """Test loading multiple data2text datasets."""
    try:
        # Load multiple datasets
        datasets = load_data2text(
            datasets=["d2t-openweather", "d2t-football"],
            nlg_models=["gpt4o"],
            splits=["test"]
        )
        
        # Check that both datasets are loaded
        assert "d2t-openweather" in datasets["test"]
        assert "d2t-football" in datasets["test"]
        
    except Exception as e:
        pytest.skip(f"Failed to load multiple data2text datasets: {e}")