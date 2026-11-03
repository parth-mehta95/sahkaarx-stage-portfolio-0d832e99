# Week 02: Expand Data Pipeline with Type-Hinted Functions

A production-grade, modular Python data processing pipeline providing reusable functions for dataset validation, transformation, and aggregation with strict type hints, explicit type contracts, and comprehensive test coverage.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Type Checked](https://img.shields.io/badge/type--checked-mypy--strict-green.svg)](http://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-84%20passed-brightgreen.svg)](https://docs.pytest.org/)

---

## Scenario

Data teams frequently require composable, robust processing operations for raw tabular data—spanning rigorous schema verification, conditional row filtering, numerical feature normalization, dynamic type coercion, and group-level aggregations. 

Ambiguous data schemas and implicit types often lead to silent data corruption, runtime crashes, and fragile pipelines. This module (`data_processor.py`) solves these issues by establishing explicit, compile-time and runtime contracts using standard Python type annotations (`PEP 484`, `PEP 526`, `PEP 585`), structural typing (`Sequence`, `Mapping`), and defensive data handling.

---

## Deliverables & Artifacts

- **`data_processor.py`**: The core modular Python module implementing type-hinted validation, transformation, aggregation, and pipeline orchestration functions.
- **`test_data_processor.py`**: Automated test suite with 30 dedicated unit tests validating type hint contracts, edge cases, error conditions, and end-to-end data pipelines.
- **`README.md`**: Complete documentation covering architecture, function signatures, input/output contracts, and execution guides.

---

## Architecture & Modular Separation of Concerns

The module is structured into four distinct, logical sections:

```
┌─────────────────────────────────────────────────────────────┐
│                       data_processor                        │
├──────────────────────────────┬──────────────────────────────┤
│ 1. Validation Group          │ 2. Transformation Group      │
│  - validate_schema()         │  - filter_by_condition()     │
│  - validate_column_ranges()  │  - normalize_numeric_columns │
│                              │  - cast_column_types()       │
├──────────────────────────────┼──────────────────────────────┤
│ 3. Aggregation Group         │ 4. Pipeline Orchestration    │
│  - aggregate_by_group()      │  - run_pipeline()            │
│  - calculate_summary_stats() │                              │
│  - pivot_and_aggregate()     │                              │
└──────────────────────────────┴──────────────────────────────┘
```

---

## Function Signatures & Type Contracts

### 1. Validation Group

#### `validate_schema()`
Ensures a DataFrame contains all required columns, matches specified Python types or pandas dtypes, and respects non-emptiness constraints.

```python
def validate_schema(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    column_types: Optional[Mapping[str, Union[type, str]]] = None,
    allow_empty: bool = False,
) -> bool:
```

- **Parameters**:
  - `df` (`pd.DataFrame`): The target DataFrame to validate.
  - `required_columns` (`Sequence[str]`): Sequence of column names that must exist in `df`.
  - `column_types` (`Optional[Mapping[str, Union[type, str]]]`, default `None`): Mapping of column name to expected Python type (`int`, `float`, `str`, `bool`) or pandas dtype string (`'int64'`, `'float64'`, `'object'`, etc.).
  - `allow_empty` (`bool`, default `False`): Whether an empty DataFrame is accepted as valid.
- **Return Value**:
  - `bool`: Returns `True` if all schema rules are satisfied.
- **Guarantees & Error Handling**:
  - Raises `TypeError` if `df` is not a `pd.DataFrame` or if any column type mismatches.
  - Raises `ValueError` if required columns are missing or if `df` is empty when `allow_empty=False`.

#### `validate_column_ranges()`
Verifies that numeric column values fall strictly within configured lower and upper boundaries.

```python
def validate_column_ranges(
    df: pd.DataFrame,
    bounds: Mapping[str, Tuple[Optional[float], Optional[float]]],
) -> Dict[str, bool]:
```

- **Parameters**:
  - `df` (`pd.DataFrame`): The target DataFrame.
  - `bounds` (`Mapping[str, Tuple[Optional[float], Optional[float]]]`): Mapping of column names to `(min_val, max_val)` bounds tuples. Set either bound to `None` to leave that side unrestricted.
- **Return Value**:
  - `Dict[str, bool]`: Per-column pass (`True`) or fail (`False`) flags indicating whether all non-null values comply.
- **Guarantees & Error Handling**:
  - Raises `TypeError` if `df` is not a `pd.DataFrame` or if target column is non-numeric.
  - Raises `KeyError` if any column specified in `bounds` does not exist.
  - Raises `ValueError` if `min_val > max_val` or if `bounds` is empty.

---

### 2. Transformation Group

#### `filter_by_condition()`
Filters rows using standardized comparison operators, returning a clean defensive copy with a reset index.

```python
def filter_by_condition(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value: Any,
) -> pd.DataFrame:
```

- **Parameters**:
  - `df` (`pd.DataFrame`): The source DataFrame.
  - `column` (`str`): Target column name to evaluate.
  - `operator` (`str`): Comparison operator (`'=='`, `'!='`, `'>'`, `'>='`, `'<'`, `'<='`, `'in'`, `'not in'`).
  - `value` (`Any`): The value or iterable collection to compare against.
- **Return Value**:
  - `pd.DataFrame`: Filtered DataFrame copy with reset indexing (`reset_index(drop=True)`).
- **Guarantees & Error Handling**:
  - Does not mutate input DataFrame (non-destructive).
  - Raises `KeyError` if `column` does not exist.
  - Raises `ValueError` for unsupported operators or if non-iterable values are provided to `'in'`/`'not in'`.

#### `normalize_numeric_columns()`
Performs mathematical scaling on numeric columns using either min-max feature scaling or z-score standardization.

```python
def normalize_numeric_columns(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    method: str = "minmax",
) -> pd.DataFrame:
```

- **Parameters**:
  - `df` (`pd.DataFrame`): The source DataFrame.
  - `columns` (`Optional[Sequence[str]]`, default `None`): Subset of numeric columns to scale. When `None`, all numeric columns are automatically detected and scaled.
  - `method` (`str`, default `"minmax"`): Scaling method—either `"minmax"` (scales to `[0.0, 1.0]`) or `"zscore"` (standardizes to `mean=0.0`, `std=1.0`).
- **Return Value**:
  - `pd.DataFrame`: Transformed DataFrame copy with scaled columns.
- **Guarantees & Error Handling**:
  - Zero-variance / constant columns are safely handled with `0.0` output to avoid division-by-zero or `NaN` outputs.
  - Raises `TypeError` if target columns are non-numeric.
  - Raises `ValueError` for invalid methods.

#### `cast_column_types()`
Safely coerces column data types with validation and descriptive error reporting.

```python
def cast_column_types(
    df: pd.DataFrame,
    type_map: Mapping[str, str],
) -> pd.DataFrame:
```

- **Parameters**:
  - `df` (`pd.DataFrame`): The input DataFrame.
  - `type_map` (`Mapping[str, str]`): Column-to-type mapping (e.g., `{'order_id': 'int64', 'timestamp': 'datetime'}`).
- **Return Value**:
  - `pd.DataFrame`: DataFrame copy with updated column types.
- **Guarantees & Error Handling**:
  - Automatically routes datetime types through `pd.to_datetime`.
  - Raises `KeyError` if any column in `type_map` is absent.
  - Raises `ValueError` with root cause explanation if casting fails.

---

### 3. Aggregation Group

#### `aggregate_by_group()`
Groups a dataset by one or multiple dimensions and aggregates target columns across one or multiple operations, producing clean, flattened headers.

```python
def aggregate_by_group(
    df: pd.DataFrame,
    group_by: Union[str, Sequence[str]],
    aggregations: Mapping[str, Union[str, Sequence[str]]],
    reset_index: bool = True,
) -> pd.DataFrame:
```

- **Parameters**:
  - `df` (`pd.DataFrame`): Input DataFrame.
  - `group_by` (`Union[str, Sequence[str]]`): Grouping column key(s).
  - `aggregations` (`Mapping[str, Union[str, Sequence[str]]]`): Mapping of column to aggregation functions (e.g., `{'revenue': ['sum', 'mean'], 'units': 'sum'}`).
  - `reset_index` (`bool`, default `True`): Whether to reset the index so grouping keys remain standard DataFrame columns.
- **Return Value**:
  - `pd.DataFrame`: Aggregated DataFrame with flattened, intuitive column headers (e.g. `revenue_sum`, `revenue_mean`).
- **Guarantees & Error Handling**:
  - Raises `KeyError` if grouping or aggregation target columns are absent.
  - Raises `ValueError` if `group_by` or `aggregations` is empty.

#### `calculate_summary_statistics()`
Generates comprehensive statistical profiles across numeric features.

```python
def calculate_summary_statistics(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
```

- **Parameters**:
  - `df` (`pd.DataFrame`): The DataFrame to analyze.
  - `columns` (`Optional[Sequence[str]]`, default `None`): Explicit list of numeric columns. If `None`, all numeric columns are evaluated.
- **Return Value**:
  - `Dict[str, Dict[str, float]]`: Nested dictionary of metric values: `count`, `mean`, `std`, `min`, `median`, and `max`.

#### `pivot_and_aggregate()`
Pivots tabular datasets by index, column headers, and values using a specified aggregation function.

```python
def pivot_and_aggregate(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: str = "sum",
    fill_value: Optional[float] = 0.0,
) -> pd.DataFrame:
```

- **Parameters**:
  - `df` (`pd.DataFrame`): Input DataFrame.
  - `index` (`str`): Column to form the pivot table rows.
  - `columns` (`str`): Column to form the pivot table header categories.
  - `values` (`str`): Numeric column to aggregate.
  - `aggfunc` (`str`, default `"sum"`): Aggregation operator.
  - `fill_value` (`Optional[float]`, default `0.0`): Missing cell replacement value.
- **Return Value**:
  - `pd.DataFrame`: Clean, un-nested pivot table with reset indices.

---

### 4. Pipeline Orchestrator

#### `run_pipeline()`
Composes schema validation, conditional filtering, numerical normalization, and group aggregation into a single, cohesive workflow.

```python
def run_pipeline(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    filter_column: Optional[str] = None,
    filter_op: Optional[str] = None,
    filter_value: Optional[Any] = None,
    normalize_columns: Optional[Sequence[str]] = None,
    normalize_method: str = "minmax",
    group_by: Optional[Union[str, Sequence[str]]] = None,
    aggregations: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
) -> pd.DataFrame:
```

---

## End-to-End Usage Example

```python
import pandas as pd
from data_processor import (
    validate_schema,
    validate_column_ranges,
    filter_by_condition,
    normalize_numeric_columns,
    aggregate_by_group,
    calculate_summary_statistics,
    run_pipeline,
)

# 1. Sample Transaction Data
raw_data = pd.DataFrame({
    "order_id": [101, 102, 103, 104, 105],
    "category": ["Electronics", "Books", "Electronics", "Home", "Books"],
    "region": ["North", "South", "North", "West", "East"],
    "units": [5, 2, 8, 1, 4],
    "unit_price": [120.0, 15.5, 95.0, 45.0, 22.0],
    "revenue": [600.0, 31.0, 760.0, 45.0, 88.0],
})

# 2. Schema and Boundary Validation
validate_schema(
    df=raw_data,
    required_columns=["order_id", "category", "units", "revenue"],
    column_types={"order_id": int, "category": str, "revenue": float},
)

range_report = validate_column_ranges(
    df=raw_data,
    bounds={"units": (1.0, 20.0), "revenue": (0.0, 10000.0)},
)
print(f"Validation bounds check: {range_report}")

# 3. Filtering & Normalization
active_orders = filter_by_condition(raw_data, "units", ">=", 2)
scaled_orders = normalize_numeric_columns(active_orders, columns=["unit_price"], method="minmax")

# 4. Group Aggregation
category_summary = aggregate_by_group(
    df=scaled_orders,
    group_by="category",
    aggregations={"revenue": ["sum", "mean"], "units": "sum"},
)
print(category_summary)

# 5. Full Pipeline Execution
pipeline_output = run_pipeline(
    df=raw_data,
    required_columns=["order_id", "category", "units", "revenue"],
    filter_column="revenue",
    filter_op=">",
    filter_value=40.0,
    normalize_columns=["unit_price"],
    group_by="category",
    aggregations={"revenue": "sum", "units": "mean"},
)
print(pipeline_output)
```

---

## Verification & Static Type Checking

### 1. Static Type Checking with Mypy (Strict Mode)

Run Mypy in strict mode to guarantee zero type ambiguities:

```bash
python -m mypy --strict data_processor.py
```

Expected output:
```
Success: no issues found in 1 source file
```

### 2. Automated Unit Tests with Pytest

Execute the automated test suite verifying all 84 test cases across the portfolio:

```bash
python -m pytest -v
```

Expected output:
```
============================= 84 passed in 1.26s ==============================
```
