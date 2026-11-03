# SahkaarX Learning Portfolio: Type-Hinted Data Pipeline Architecture

A modular, production-ready Python data pipeline repository built with clear type contracts, PEP-compliant type annotations, and automated test coverage across loading, validation, transformation, aggregation, and object-oriented pipeline orchestration.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Type Checked](https://img.shields.io/badge/type--checked-mypy--strict-green.svg)](http://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-208%20passed-brightgreen.svg)](https://docs.pytest.org/)

---

## Project Structure

```
├── data_loader.py                 # Week 01 & 03: CSV & JSON DataLoaders (BaseDataLoader, DataLoader, JSONDataLoader)
├── data_processor.py              # Week 02, 03 & 04: OOP & Functional Processors & Validators
├── pipeline.py                    # Week 03 & 04: OOP Pipeline Orchestrator & Multi-Constraint Validation
├── pytest.ini                     # Global pytest test runner configuration
├── README.md                      # Repository overview & architectural documentation
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py        # Automated test suite for data_loader (12 tests)
│   ├── test_data_processor.py     # Automated test suite for data_processor (30 tests)
│   ├── test_data_validation.py    # Automated test suite for data validation & constraints (35 tests)
│   └── test_oop_pipeline.py       # Automated test suite for OOP classes & pipeline (27 tests)
└── practice/
    ├── week-01-build-type-hinted-data-pipeline-module/
    │   ├── data_loader.py         # Week 01 module copy
    │   ├── test_data_loader.py    # Week 01 test suite copy (12 tests)
    │   └── README.md              # Week 01 task documentation
    ├── week-02-expand-data-pipeline-with-type-hinted-functions/
    │   ├── data_processor.py      # Week 02 module copy
    │   ├── test_data_processor.py # Week 02 test suite copy (30 tests)
    │   └── README.md              # Week 02 task brief & contract documentation
    ├── week-03-practice-python-programming/
    │   ├── data_loader.py         # Week 03 OOP & functional loader copy
    │   ├── data_processor.py      # Week 03 OOP & functional processor copy
    │   ├── pipeline.py            # Week 03 OOP pipeline orchestrator copy
    │   ├── test_oop_pipeline.py   # Week 03 test suite copy (27 tests)
    │   └── README.md              # Week 03 architectural & OOP documentation
    └── week-04-practice-version-control/
        ├── data_loader.py         # Week 04 module copy
        ├── data_processor.py      # Week 04 module copy with validation functions
        ├── pipeline.py            # Week 04 pipeline module copy with validation
        ├── test_data_validation.py# Week 04 test suite copy (35 tests)
        └── README.md              # Week 04 Git workflow, PR guide & validation docs
```

---

## Module Overview

### 1. `data_loader.py` (Week 01 & Week 03)
Reusable utilities and OOP classes for loading tabular files (CSV, JSON) into pandas DataFrames and performing base cleaning with static type safety.

#### Object-Oriented Classes
- **`BaseDataLoader` (ABC)**: Abstract base class defining `load()` and `save()` contracts with path validation, state caching (`data`, `is_loaded`), data cleaning (`clean()`), and metadata introspection (`get_summary()`).
- **`DataLoader` (CSVDataLoader)**: Concrete CSV data loader with configurable delimiters and encodings.
- **`JSONDataLoader`**: Concrete JSON data loader demonstrating inheritance and polymorphism.

#### Functional Utilities
- **`load_csv(file_path: Union[str, Path], delimiter: str = ",", encoding: str = "utf-8", **kwargs: Any) -> pd.DataFrame`**: Validates file existence and loads tabular data.
- **`transform_data(df: pd.DataFrame, drop_na: bool = True, drop_duplicates: bool = False, strip_whitespace: bool = True, fill_na_value: Optional[Any] = None) -> pd.DataFrame`**: Cleans whitespace, drops/fills nulls, and deduplicates records defensively.
- **`save_csv(df: pd.DataFrame, file_path: Union[str, Path], index: bool = False, encoding: str = "utf-8", **kwargs: Any) -> Path`**: Exports DataFrame to CSV, ensuring directory creation.
- **`get_data_summary(df: pd.DataFrame) -> Dict[str, Any]`**: Generates shape, columns, null counts, and data types summary.

---

### 2. `data_processor.py` (Week 02 & Week 03)
Modular component extending the data pipeline with composable validation, transformation, and aggregation functions and OOP classes.

#### Object-Oriented Classes
- **`BaseDataProcessor` (ABC)**: Abstract base class enforcing `process(df: pd.DataFrame) -> pd.DataFrame` with audit logging and callable syntax `__call__`.
- **`DataValidator`**: Focuses on schema constraints and column numeric boundary checks.
- **`DataTransformer`**: Focuses on conditional filtering, feature normalization, and type casting.
- **`DataAggregator`**: Focuses on multi-metric group aggregations, summary statistics, and pivot tables.
- **`DataProcessor`**: Comprehensive processor providing a fluent chaining API (`set_data()`, `filter()`, `normalize()`, `cast()`, `get_data()`) and organized method namespaces.

#### Functional Utilities
- **Validation**: `validate_schema()`, `validate_column_ranges()`, `validate_null_constraints()`, `validate_unique_constraints()`, `validate_allowed_values()`
- **Transformation**: `filter_by_condition()`, `normalize_numeric_columns()`, `cast_column_types()`
- **Aggregation**: `aggregate_by_group()`, `calculate_summary_statistics()`, `pivot_and_aggregate()`
- **Orchestration**: `run_pipeline()`

---

### 3. `pipeline.py` (Week 03 & Week 04)
Object-oriented workflow orchestrator executing multi-stage data processing with telemetry and data validation.

- **`BasePipeline` (ABC)**: Abstract interface for stage registration and execution.
- **`DataPipeline`**: Chains a `BaseDataLoader` with sequential `BaseDataProcessor` stages, tracking execution time, input shapes, and output shapes. Includes `.validate()` for telemetry diagnostics and `.add_validator()` for fluent validation stage registration.
- **`validate_pipeline_data()`**: Multi-constraint tabular data validation orchestrator verifying schema, missing ratios, uniqueness keys, numeric boundaries, and categorical allowed values.
- **`build_pipeline(...)` & `create_standard_pipeline(...)`**: Factory helper functions for rapid pipeline configuration with built-in validation support.

---

## Benefits of Type Hints & OOP Architecture

1. **Explicit API Contracts**: Eliminates guesswork around accepted parameter shapes and return formats.
2. **Compile-Time Static Type Safety**: `mypy --strict` detects incompatible operations and potential null/type mismatches before code runs.
3. **Inheritance & Polymorphism**: Common abstractions (`BaseDataLoader`, `BaseDataProcessor`) allow swapping CSV for JSON or adding custom processor steps interchangeably.
4. **Fluent Expressiveness & Encapsulation**: Method chaining with internal state reduces boilerplate.
5. **Telemetry & Auditability**: Step-level duration and shape tracking provides transparency into pipeline health.

---

## Installation & Quickstart

```bash
# Install required dependencies
pip install pandas numpy mypy pytest
```

### Example Usage: Object-Oriented Pipeline with Data Validation

```python
import pandas as pd
from data_loader import DataLoader
from data_processor import DataTransformer, DataAggregator
from pipeline import DataPipeline

# 1. Initialize pipeline with CSV loader, validation constraints, and processor stages
pipeline = (
    DataPipeline(loader=DataLoader(file_path="data/employees.csv"))
    .add_validator(
        required_columns=["department", "salary", "experience"],
        non_null_columns=["department", "salary"],
        unique_columns="employee_id",
        bounds={"salary": (20000.0, 500000.0)},
    )
    .add_processor(DataTransformer(filters=[("experience", ">=", 3)], normalize_cols=["salary"]))
    .add_processor(DataAggregator(group_by="department", aggregations={"salary": ["mean", "count"]}))
)

# 2. Execute pipeline
results_df = pipeline.execute()
print(results_df)

# 3. View telemetry summary
for step in pipeline.execution_summary:
    print(step)
```

---

## Verification & Testing

### Static Type Checking with Mypy (Strict Mode)

```bash
python -m mypy --strict data_loader.py data_processor.py pipeline.py
```

Output:
```
Success: no issues found in 3 source files
```

### Running Automated Test Suite with Pytest

```bash
python -m pytest -v
```

All 208 automated unit tests pass across `tests/` and `practice/` suites:
- 24 tests for `data_loader` (12 root + 12 practice copy)
- 60 tests for `data_processor` (30 root + 30 practice copy)
- 70 tests for `data_validation` (35 root + 35 practice copy)
- 54 tests for `oop_pipeline` (27 root + 27 practice copy)
