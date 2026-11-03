# SahkaarX Learning Portfolio: Type-Hinted Data Pipeline Architecture

A modular, production-ready Python data pipeline repository built with clear type contracts, PEP-compliant type annotations, and automated test coverage across loading, validation, transformation, and aggregation modules.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Type Checked](https://img.shields.io/badge/type--checked-mypy--strict-green.svg)](http://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-84%20passed-brightgreen.svg)](https://docs.pytest.org/)

---

## Project Structure

```
├── data_loader.py                 # Week 01: Core CSV loading & cleaning module
├── data_processor.py              # Week 02: Validation, transformation & aggregation module
├── pytest.ini                     # Global pytest test runner configuration
├── README.md                      # Repository overview & architectural documentation
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py        # Automated test suite for data_loader (12 tests)
│   └── test_data_processor.py     # Automated test suite for data_processor (30 tests)
└── practice/
    ├── week-01-build-type-hinted-data-pipeline-module/
    │   ├── data_loader.py         # Week 01 module copy
    │   ├── test_data_loader.py    # Week 01 test suite copy
    │   └── README.md              # Week 01 task documentation
    └── week-02-expand-data-pipeline-with-type-hinted-functions/
        ├── data_processor.py      # Week 02 module copy
        ├── test_data_processor.py # Week 02 test suite copy
        └── README.md              # Week 02 task brief & contract documentation
```

---

## Module Overview

### 1. `data_loader.py` (Week 01)
Reusable utilities for loading CSV files into pandas DataFrames and performing base cleaning with static type safety.

- **`load_csv(file_path: Union[str, Path], delimiter: str = ",", encoding: str = "utf-8", **kwargs: Any) -> pd.DataFrame`**: Validates file existence and loads tabular data.
- **`transform_data(df: pd.DataFrame, drop_na: bool = True, drop_duplicates: bool = False, strip_whitespace: bool = True, fill_na_value: Optional[Any] = None) -> pd.DataFrame`**: Cleans whitespace, drops/fills nulls, and deduplicates records defensively.
- **`save_csv(df: pd.DataFrame, file_path: Union[str, Path], index: bool = False, encoding: str = "utf-8", **kwargs: Any) -> Path`**: Exports DataFrame to CSV, ensuring directory creation.
- **`get_data_summary(df: pd.DataFrame) -> Dict[str, Any]`**: Generates shape, columns, null counts, and data types summary.

---

### 2. `data_processor.py` (Week 02)
Modular component extending the data pipeline with composable validation, transformation, and aggregation functions.

#### Validation Functions
- **`validate_schema(df: pd.DataFrame, required_columns: Sequence[str], column_types: Optional[Mapping[str, Union[type, str]]] = None, allow_empty: bool = False) -> bool`**:
  Verifies required column presence, column data types, and non-emptiness constraints.
- **`validate_column_ranges(df: pd.DataFrame, bounds: Mapping[str, Tuple[Optional[float], Optional[float]]]) -> Dict[str, bool]`**:
  Checks that numeric values fall within specified `[min, max]` intervals.

#### Transformation Functions
- **`filter_by_condition(df: pd.DataFrame, column: str, operator: str, value: Any) -> pd.DataFrame`**:
  Filters rows using comparison operators (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not in`) on defensive copies.
- **`normalize_numeric_columns(df: pd.DataFrame, columns: Optional[Sequence[str]] = None, method: str = "minmax") -> pd.DataFrame`**:
  Scales numeric columns using min-max normalization (`[0.0, 1.0]`) or z-score standardization (`mean=0`, `std=1`), handling zero variance safely.
- **`cast_column_types(df: pd.DataFrame, type_map: Mapping[str, str]) -> pd.DataFrame`**:
  Safely coerces column data types with descriptive error messages.

#### Aggregation Functions
- **`aggregate_by_group(df: pd.DataFrame, group_by: Union[str, Sequence[str]], aggregations: Mapping[str, Union[str, Sequence[str]]], reset_index: bool = True) -> pd.DataFrame`**:
  Groups data across one or multiple dimensions and aggregates target columns, generating flattened headers.
- **`calculate_summary_statistics(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]`**:
  Computes `count`, `mean`, `std`, `min`, `median`, and `max` for numeric columns.
- **`pivot_and_aggregate(df: pd.DataFrame, index: str, columns: str, values: str, aggfunc: str = "sum", fill_value: Optional[float] = 0.0) -> pd.DataFrame`**:
  Pivots tabular datasets into structured aggregated matrices.

#### Pipeline Orchestrator
- **`run_pipeline(...) -> pd.DataFrame`**:
  Chains schema validation, conditional row filtering, numerical scaling, and grouping into an integrated workflow.

---

## Benefits of Type Hints & Static Contracts

1. **Explicit API Contracts**: Eliminates guesswork around accepted parameter shapes and return formats.
2. **Compile-Time Static Type Safety**: `mypy --strict` detects incompatible operations and potential null/type mismatches before code runs.
3. **IDE Productivity**: Real-time autocompletion, instant parameter inspection, and automated refactoring support.
4. **Resilient Data Pipelines**: Type contracts protect against downstream pipeline breakages due to schema drifts.
5. **Executable Self-Documentation**: Type annotations serve as live, machine-verifiable documentation.

---

## Installation & Quickstart

```bash
# Install required dependencies
pip install pandas numpy mypy pytest
```

### Example Usage

```python
import pandas as pd
from data_processor import validate_schema, filter_by_condition, aggregate_by_group

# Sample data
df = pd.DataFrame({
    "department": ["Engineering", "Engineering", "Marketing", "HR"],
    "salary": [110000.0, 125000.0, 95000.0, 85000.0],
    "experience_years": [4, 6, 3, 2],
})

# 1. Validate schema
validate_schema(df, required_columns=["department", "salary", "experience_years"])

# 2. Filter high-salary employees
experienced_df = filter_by_condition(df, "experience_years", ">=", 3)

# 3. Aggregate metrics by department
dept_summary = aggregate_by_group(
    experienced_df,
    group_by="department",
    aggregations={"salary": ["mean", "sum"]},
)
print(dept_summary)
```

---

## Verification & Testing

### Static Type Checking with Mypy (Strict Mode)

```bash
python -m mypy --strict data_processor.py
```

Output:
```
Success: no issues found in 1 source file
```

### Running Automated Test Suite with Pytest

```bash
python -m pytest -v
```

All 84 automated unit tests pass across `tests/` and `practice/` suites:
- 12 tests for `data_loader` (loading, transforms, error handling, type contracts)
- 30 tests for `data_processor` (schema validation, boundary checks, filters, scaling, grouping, pivot, pipeline, type contracts)
- 42 tests across identical practice directory copies
