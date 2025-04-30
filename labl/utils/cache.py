import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import requests
from transformers.utils.import_utils import is_pandas_available

if TYPE_CHECKING:
    if is_pandas_available():
        import pandas as pd


DEFAULT_XDG_CACHE = os.path.expanduser("~/.cache")
XDG_CACHE = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE)
DEFAULT_LABL_CACHE = os.path.join(XDG_CACHE, "labl")
LABL_CACHE = os.path.expanduser(os.getenv("LABL_CACHE", DEFAULT_LABL_CACHE))
LABL_DATASETS_CACHE = os.path.join(LABL_CACHE, "datasets")


def load_cached_or_download(
    url: str,
    filetype: Literal["json", "jsonl"],
    cache_path: str = LABL_DATASETS_CACHE,
) -> "pd.DataFrame":
    """
    Loads a file from the cache folder if found, or downloads it if missing.

    Args:
        url (str): The URL of the file to load.
        filetype (Literal["json", "jsonl"]): The type of the file to load.
        cache_path (str): The path to the cache folder. Defaults to `LABL_DATASETS_CACHE`.
    """
    if not is_pandas_available():
        raise RuntimeError("The `pandas` library is not installed. Please install it to use this function.")

    import pandas as pd

    cache_folder = Path(cache_path)
    cache_file = cache_folder / os.path.basename(url)
    if not cache_file.exists():
        logging.info(f"Cache file '{cache_file}' not found. Downloading from {url}...")
        # If the file doesn't exist, download it
        try:
            response = requests.get(url)
            response.raise_for_status()
            cache_folder.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                f.write(response.content)
            logging.info(f"Downloaded dataset from {url} to {cache_file}")
        except requests.RequestException as e:
            logging.error(f"Failed to download dataset from URL '{url}': {e}")
            raise
    if filetype in ["json", "jsonl"]:
        return pd.read_json(cache_file, lines=filetype == "jsonl")
    else:
        raise ValueError(f"Unsupported file type: {filetype}. Supported types are 'json' and 'jsonl'.")
