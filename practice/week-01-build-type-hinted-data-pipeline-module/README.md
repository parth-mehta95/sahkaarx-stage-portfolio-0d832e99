# Week 01: Build Type-Hinted Data Pipeline Module

A reusable, production-ready Python data loader and preprocessing module developed with strict type annotations, static type safety, and clear type contracts.

---

## Scenario

Data teams frequently encounter issues when loading and preprocessing raw CSV files due to ambiguity in function input/output structures. This module (`data_loader.py`) addresses those challenges by providing reusable, type-annotated functions for:
1. Loading CSV datasets from file system paths (`load_csv`).
2. Cleansing, sanitizing, and transforming tabular datasets (`transform_data`).
3. Exporting processed data and generating structural schema summaries (`save_csv`, `get_data_summary`).

---

## Deliverables & Artifacts

- **`data_loader.py`**: The core Python module with type-annotated functions.
- **`test_data_loader.py`**: Automated test suite validating data operations and type contracts.
- **`README.md`**: Comprehensive documentation explaining function signatures, type hints, and architectural benefits.

---

## Type Contracts & Function Signatures

### 1. `load_csv()`

```python
def load_csv(
    file_path: Union[str, Path],
    delimiter: str = ",",
    encoding: str = "utf-8",
    **kwargs: Any,
) -> pd.DataFrame:
```

- **Parameters**:
  - `file_path` (`Union[str, Path]`): Accepts either a standard string file path or a `pathlib.Path` object. This contract guarantees path flexibility across operating systems.
  - `delimiter` (`str`, default `","`): Field separator character.
  - `encoding` (`str`, default `"utf-8"`): File character encoding.
  - `**kwargs` (`Any`): Optional keyword arguments forwarded to the underlying pandas CSV reader.
- **Return Value**:
  - `pd.DataFrame`: Strictly returns a populated pandas DataFrame.
- **Guarantees & Error Handling**:
  - Raises `ValueError` if `file_path` is empty or resolves to a directory.
  - Raises `FileNotFoundError` if the specified file does not exist.

---

### 2. `transform_data()`

```python
def transform_data(
    df: pd.DataFrame,
    drop_na: bool = True,
    drop_duplicates: bool = False,
    strip_whitespace: bool = True,
    fill_na_value: Optional[Any] = None,
) -> pd.DataFrame:
```

- **Parameters**:
  - `df` (`pd.DataFrame`): The source pandas DataFrame to process.
  - `drop_na` (`bool`, default `True`): Flag indicating whether rows containing missing/null values should be removed.
  - `drop_duplicates` (`bool`, default `False`): Flag indicating whether duplicate rows should be deduplicated.
  - `strip_whitespace` (`bool`, default `True`): Flag indicating whether string columns and column names should be trimmed of leading/trailing whitespace. Empty strings are mapped to null (`pd.NA`) to enable uniform missing value handling.
  - `fill_na_value` (`Optional[Any]`, default `None`): Optional replacement value for null entries when `drop_na=False`.
- **Return Value**:
  - `pd.DataFrame`: Returns a transformed copy of the DataFrame with reset indexing.
- **Guarantees**:
  - Performs non-destructive operations on a defensive copy (`df.copy()`).
  - Raises `TypeError` if input is not a DataFrame.

---

### 3. Utility Functions

- **`save_csv(df: pd.DataFrame, file_path: Union[str, Path], index: bool = False, encoding: str = "utf-8", **kwargs: Any) -> Path`**:
  Persists a DataFrame to disk, ensuring directory existence, and returns the resolved `Path`.
- **`get_data_summary(df: pd.DataFrame) -> Dict[str, Any]`**:
  Extracts shape, columns, missing value counts, and data types into a structured dictionary.

---

## Benefits of Type Hints

Adopting Python type annotations (`PEP 484`, `PEP 526`, `PEP 585`) provides critical software engineering advantages:

1. **Clear Type Contracts & API Self-Documentation**:
   Function signatures communicate input expectations and return structures directly in the code, reducing guesswork and eliminating out-of-date docstrings.
2. **Static Bug Prevention (`mypy`)**:
   Static type checkers detect invalid parameter types, missing return statements, and incompatible object manipulations before runtime execution.
3. **IDE Productivity & Autocompletion**:
   Modern editors (VS Code, PyCharm) leverage type signatures to provide intelligent auto-completion, parameter hints, and instantaneous inline linting.
4. **Confident Refactoring**:
   When changing schemas or parameter requirements across data pipelines, type annotations expose downstream breaking changes across the codebase.
5. **Team Collaboration & Code Maintainability**:
   New team members can readily understand expected data structures, accelerating onboarding and reducing code review cycles.

---

## Installation & Setup

Ensure Python 3.10+ is installed:

```bash
# Install dependencies
pip install pandas mypy pytest
```

---

## Usage Example

```python
from pathlib import Path
from data_loader import load_csv, transform_data, get_data_summary, save_csv

# 1. Load CSV
raw_df = load_csv("data/raw_dataset.csv")

# 2. Transform and clean data
cleaned_df = transform_data(
    df=raw_df,
    drop_na=True,
    drop_duplicates=True,
    strip_whitespace=True,
)

# 3. Inspect structural summary
summary = get_data_summary(cleaned_df)
print(f"Cleaned dataset: {summary['num_rows']} rows, {summary['num_columns']} columns")

# 4. Save cleaned output
output_path = save_csv(cleaned_df, Path("data/cleaned_dataset.csv"))
print(f"Saved to: {output_path}")
```

---

## Verification & Testing

### 1. Static Type Checking with Mypy

```bash
mypy data_loader.py
```

Expected output:
```
Success: no issues found in 1 source file
```

### 2. Automated Unit Tests with Pytest

```bash
python -m pytest test_data_loader.py -v
```

All unit tests pass, verifying both runtime behavior and type contracts.
