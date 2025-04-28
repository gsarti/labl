from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version
from pathlib import Path

import toml

from labl.data import EditedDataset, EditedEntry, LabeledDataset, LabeledEntry
from labl.utils import HuggingfaceTokenizer, WhitespaceTokenizer, WordBoundaryTokenizer

__package_version = "unknown"


def __get_package_version() -> str:
    """Find the version of this package."""
    global __package_version  # noqa: PLW0603

    if __package_version != "unknown":
        return __package_version
    try:
        __package_version = metadata_version("labl")
    except PackageNotFoundError:
        pyproject_toml_file = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_toml_file.exists() and pyproject_toml_file.is_file():
            __package_version = toml.load(pyproject_toml_file)["tool"]["project"]["version"]
            # Indicate it might be locally modified or unreleased.
            __package_version = __package_version + "+"

    return __package_version


def __getattr__(name: str) -> str:
    """Get package attributes."""
    if name in ("version", "__version__"):
        return __get_package_version()
    else:
        raise AttributeError(f"No attribute {name} in module {__name__}.")


__all__ = [
    "LabeledEntry",
    "EditedEntry",
    "LabeledDataset",
    "EditedDataset",
    "WhitespaceTokenizer",
    "WordBoundaryTokenizer",
    "HuggingfaceTokenizer",
]
