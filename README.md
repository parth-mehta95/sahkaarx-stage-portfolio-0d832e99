# SahkaarX Learning Portfolio: Type-Hinted Data Pipeline Module

A reusable, production-ready Python data loader and preprocessing module built with clear type contracts, PEP-compliant type annotations, and automated test coverage.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Type Checked](https://img.shields.io/badge/type--checked-mypy-green.svg)](http://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](https://docs.pytest.org/)

---

## Project Structure

```
├── data_loader.py                 # Core reusable module with type-hinted functions
├── README.md                      # Repository documentation and type hint guide
├── tests/
│   └── test_data_loader.py        # Automated test suite
└── practice/
    └── week-01-build-type-hinted-data-pipeline-module/
        ├── data_loader.py         # Module copy for Week 01 submission
        ├── test_data_loader.py    # Week 01 test suite
        └── README.md              # Week 01 task brief and documentation
```

---

## Module Overview: `data_loader.py`

This module provides reusable utilities for loading CSV files into pandas DataFrames and performing common cleaning and transformation operations with strict static type safety.

### Key Functions & Type Contracts

#### 1. `load_csv()`
```python
def load_csv(
    file_path: Union[str, Path],
    delimiter: str = ",",
    encoding: str = "utf-8",
    **kwargs: Any,
) -> pd.DataFrame:
```
- **Type Annotations**:
  - `file_path: Union[str, Path]`: Accepts string paths or `pathlib.Path` objects.
  - `delimiter: str`: Delimiter character (default: `","`).
  - `encoding: str`: File encoding (default: `"utf-8"`).
  - `**kwargs: Any`: Optional reader parameters passed to pandas.
  - Returns: `pd.DataFrame`.
- **Validation**:
  - Validates that the file exists and is not a directory.
  - Raises `FileNotFoundError` or `ValueError` on bad inputs.

#### 2. `transform_data()`
```python
def transform_data(
    df: pd.DataFrame,
    drop_na: bool = True,
    drop_duplicates: bool = False,
    strip_whitespace: bool = True,
    fill_na_value: Optional[Any] = None,
) -> pd.DataFrame:
```
- **Type Annotations**:
  - `df: pd.DataFrame`: Input DataFrame.
  - `drop_na: bool`: Drop rows with null values (default: `True`).
  - `drop_duplicates: bool`: Remove duplicate rows (default: `False`).
  - `strip_whitespace: bool`: Trim whitespace from column names and string cells, converting empty strings to nulls (default: `True`).
  - `fill_na_value: Optional[Any]`: Value used to fill nulls if `drop_na=False` (default: `None`).
  - Returns: `pd.DataFrame`.
- **Safety**:
  - Performs transformations on a defensive copy to prevent mutations of caller data.
  - Resets the index after filtering.

#### 3. Auxiliary Functions
- `save_csv(df: pd.DataFrame, file_path: Union[str, Path], index: bool = False, encoding: str = "utf-8", **kwargs: Any) -> Path`: Exports DataFrame to CSV, ensuring parent directories exist.
- `get_data_summary(df: pd.DataFrame) -> Dict[str, Any]`: Computes row count, column count, column names, null counts, and column data types.

---

## Benefits of Type Hints

1. **Clear Type Contracts**: Signatures define explicit parameter requirements and return values, preventing ambiguous function calls.
2. **Compile-Time Static Type Checking**: Tools like `mypy` detect type errors, incorrect argument types, and missing fields before runtime execution.
3. **Enhanced Developer Experience**: IDEs provide context-aware autocompletion, parameter hints, and instantaneous error detection.
4. **Resilient Refactoring**: Structural changes to data pipelines are verified across the codebase via static analyzers.
5. **Living Documentation**: Eliminates stale docstrings; the type annotations themselves act as continuously checked documentation.

---

## Getting Started

### Installation

```bash
pip install pandas mypy pytest
```

### Quick Usage

```python
from pathlib import Path
from data_loader import load_csv, transform_data, get_data_summary

# Load dataset
df = load_csv("data.csv")

# Clean and transform
clean_df = transform_data(df, drop_na=True, drop_duplicates=True)

# Inspect summary
summary = get_data_summary(clean_df)
print(summary)
```

---

## Verification & Testing

### Static Type Checking with Mypy
```bash
mypy data_loader.py
```
Output:
```
Success: no issues found in 1 source file
```

### Running Unit Tests with Pytest
```bash
python -m pytest tests/ -v
```
All tests verify function behavior, error conditions, edge cases, and type annotations via runtime inspection (`typing.get_type_hints`).
