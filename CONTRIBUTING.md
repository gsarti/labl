# Contributing to `labl` 🏷️

Thank you for your interest in contributing to `labl`, a token-level label management toolkit! This document provides guidelines and instructions for setting up the development environment and contributing to the project.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Making Contributions](#making-contributions)

## Development Environment Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://astral.sh/uv) package manager (recommended)

### Setting Up the Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/gsarti/labl.git
   cd labl
   ```

2. Download and install the uv package manager:
   ```bash
   make uv-download
   ```

3. Install development dependencies:
   ```bash
   make install-dev
   ```

   This command will:
   - Create a virtual environment
   - Install all required dependencies
   - Install pre-commit hooks
   - Install the package in development mode

4. Activate the virtual environment:
   ```bash
   # On Linux/macOS
   source .venv/bin/activate
   
   # On Windows
   .\.venv\Scripts\activate
   ```

## Project Structure

The project is organized as follows:

```
labl/
├── labl/                   # Main package source code
│   ├── commands/           # CLI commands
│   ├── data/               # Data structures and handling
│   ├── datasets/           # Dataset loaders
│   └── utils/              # Utility functions and classes
├── tests/                  # Test files
├── docs/                   # Documentation
├── examples/               # Example usage
├── Makefile                # Development commands
├── pyproject.toml          # Project configuration
├── requirements.txt        # Core dependencies
└── requirements-dev.txt    # Development dependencies
```

## Coding Standards

### Python Version

The project requires Python 3.10 or higher and is tested with Python 3.10, 3.11, and 3.12.

### Code Style

We use [ruff](https://github.com/astral-sh/ruff) for code formatting and linting. The configuration is defined in `pyproject.toml`.

Key style guidelines:
- Line length: 119 characters
- Docstring style: Google
- Type annotations: Required for all functions and methods

To check your code style without making changes:
```bash
make check-style
```

To automatically fix style issues:
```bash
make fix-style
```

### Docstrings

All public modules, functions, classes, and methods should have docstrings following the Google style:

```python
def function(arg1: str, arg2: int = 42) -> bool:
    """Short description of the function.

    Longer description if needed, explaining what the function does
    in more detail.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2. Default: 42.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When something goes wrong.
    """
```

### Type Annotations

Type annotations are required for all function parameters, return values, and class attributes. The project uses standard Python typing.

```python
from typing import Any, List, Optional, Dict

def process_data(data: List[str], config: Optional[Dict[str, Any]] = None) -> bool:
    ...
```

## Testing

Tests are written using [pytest](https://docs.pytest.org/). All new features should include tests.

To run the tests:
```bash
make test
```

Tests are located in the `tests/` directory and should follow the naming convention `test_*.py`.

## Documentation

Documentation is built using [MkDocs](https://www.mkdocs.org/) with the [Material](https://squidfunk.github.io/mkdocs-material/) theme.

To build the documentation:
```bash
make build-docs
```

To serve the documentation locally:
```bash
make serve-docs
```

To build and serve in one command:
```bash
make docs
```

### API Documentation

API documentation is automatically generated from docstrings. Make sure your docstrings are complete and follow the Google style.

## Pre-commit Hooks

The project uses [pre-commit](https://pre-commit.com/) to run checks before each commit. The hooks are configured in `.pre-commit-config.yaml` and include:

- Code formatting with ruff
- Code linting with ruff
- Running tests
- Building documentation
- Various file checks (trailing whitespace, YAML validation, etc.)

Pre-commit hooks are installed automatically when running `make install-dev`.

## Making Contributions

### Workflow

1. Create a fork of the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Run tests and ensure they pass
5. Run style checks and fix any issues
6. Submit a pull request

### Dependency Management

If you need to add new dependencies:

1. Add them to the appropriate section in `pyproject.toml`
2. Update the dependency files:
   ```bash
   make update-deps
   ```

This will update both `requirements.txt` and `requirements-dev.txt`.

### Commit Messages

Write clear, concise commit messages that explain the changes you've made. Follow these guidelines:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests where appropriate

### Pull Requests

When submitting a pull request:

1. Provide a clear description of the changes
2. Link to any related issues
3. Ensure all tests pass
4. Make sure the documentation is updated if necessary

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [Apache 2.0 License](LICENSE).