# Week 03 Practice: Python Programming - Object-Oriented Data Pipeline Architecture

A production-grade, object-oriented refactoring of the tabular data processing pipeline, introducing abstract base classes, polymorphic file loaders, specialized processing components, and an end-to-end pipeline orchestrator with strict type hints, explicit type contracts, and comprehensive unit test coverage.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Type Checked](https://img.shields.io/badge/type--checked-mypy--strict-green.svg)](http://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-138%20passed-brightgreen.svg)](https://docs.pytest.org/)

---

## Scenario & Task Brief

As a Data Scientist contributing to the **"Data Pipeline & Repository"** capstone, this milestone focuses on refactoring procedural data processing functions into robust, reusable Object-Oriented Programming (OOP) classes (`DataLoader`, `DataProcessor`, `DataPipeline`) with proper inheritance hierarchies, polymorphic interfaces, and clean method organization.

Building directly upon prior milestones ([Week 01 CSV Loader](https://github.com/parth-mehta95/sahkaarx-stage-portfolio-0d832e99) and [Week 02 Type-Hinted Data Processor](https://github.com/parth-mehta95/sahkaarx-stage-portfolio-0d832e99)), this milestone establishes:
1. **Abstract Base Classes (`abc.ABC`)** enforcing uniform contracts across components (`BaseDataLoader`, `BaseDataProcessor`, `BasePipeline`).
2. **Specialized & Unified Subclasses** (`DataValidator`, `DataTransformer`, `DataAggregator`, `DataProcessor`, `DataLoader`, `JSONDataLoader`).
3. **Fluent Method Chaining & State Management** allowing expressive data manipulation pipelines (`processor.set_data(df).filter(...).normalize(...).cast(...)`).
4. **End-to-End Orchestration** with `DataPipeline` capturing step-level execution telemetry (durations, input/output shapes, audit logs).
5. **Zero Breaking Changes**: 100% backward compatibility preserved for functional APIs (`load_csv`, `transform_data`, `save_csv`, `validate_schema`, `aggregate_by_group`, etc.).

---

## Deliverables & Artifacts

- **`data_loader.py`**: Refactored loader module containing `BaseDataLoader` (ABC), `DataLoader` (CSV), and `JSONDataLoader` alongside legacy functional utilities.
- **`data_processor.py`**: Refactored processing module containing `BaseDataProcessor` (ABC), `DataValidator`, `DataTransformer`, `DataAggregator`, and `DataProcessor` with organized method categories.
- **`pipeline.py`**: Object-Oriented pipeline orchestrator module defining `BasePipeline` (ABC), `DataPipeline`, and factory functions (`build_pipeline`, `create_standard_pipeline`).
- **`test_oop_pipeline.py`**: Automated unit test suite featuring 27 dedicated tests validating OOP contracts, inheritance, polymorphism, method chaining, telemetry, and error conditions.
- **`README.md`**: Architectural documentation, class hierarchy diagrams, type contract breakdowns, and execution walkthroughs.

---

## Architecture & Class Hierarchies

### 1. Data Loader Hierarchy

```
                    ┌────────────────────────────┐
                    │       BaseDataLoader       │ (ABC)
                    ├────────────────────────────┤
                    │ + file_path: Optional[Path]│
                    │ + data: Optional[DataFrame]│
                    │ + is_loaded: bool          │
                    ├────────────────────────────┤
                    │ + load() [abstract]        │
                    │ + save() [abstract]        │
                    │ + clean()                  │
                    │ + get_summary()            │
                    └─────────────┬──────────────┘
                                  │
           ┌──────────────────────┴──────────────────────┐
           ▼                                             ▼
┌────────────────────────────┐                ┌────────────────────────────┐
│         DataLoader         │                │       JSONDataLoader       │
│      (CSVDataLoader)       │                │                            │
├────────────────────────────┤                ├────────────────────────────┤
│ + delimiter: str           │                │ + orient: Optional[str]    │
│ + encoding: str            │                │ + encoding: str            │
├────────────────────────────┤                ├────────────────────────────┤
│ + load(source, ...)        │                │ + load(source, ...)        │
│ + save(df, destination, ..)│                │ + save(df, destination, ..)│
└────────────────────────────┘                └────────────────────────────┘
```

### 2. Data Processor Hierarchy

```
                    ┌────────────────────────────┐
                    │     BaseDataProcessor      │ (ABC)
                    ├────────────────────────────┤
                    │ + name: str                │
                    │ + audit_log: List[Dict]    │
                    ├────────────────────────────┤
                    │ + process(df) [abstract]   │
                    │ + __call__(df)             │
                    │ + log_step()               │
                    │ + reset_audit_log()        │
                    └─────────────┬──────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  DataValidator   │    │ DataTransformer  │    │  DataAggregator  │
├──────────────────┤    ├──────────────────┤    ├──────────────────┤
│ + validate_schema│    │ + filter_by_cond │    │ + aggregate_group│
│ + validate_ranges│    │ + normalize_cols │    │ + summary_stats  │
│ + process()      │    │ + cast_types     │    │ + pivot_aggregate│
└──────────────────┘    │ + add_filter()   │    │ + process()      │
                        │ + process()      │    └──────────────────┘
                        └──────────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │       DataProcessor        │
                    ├────────────────────────────┤
                    │ + _df: Optional[DataFrame] │
                    │ + validator: DataValidator │
                    │ + trans: DataTransformer   │
                    │ + agg: DataAggregator      │
                    ├────────────────────────────┤
                    │ [Fluent Chaining API]      │
                    │ + set_data(df) -> self     │
                    │ + filter(...) -> self      │
                    │ + normalize(...) -> self   │
                    │ + cast(...) -> self        │
                    │ + get_data() -> DataFrame  │
                    │ [Organized Method Groups]  │
                    │ + validate_schema()        │
                    │ + validate_column_ranges() │
                    │ + filter_by_condition()    │
                    │ + normalize_numeric_cols() │
                    │ + cast_column_types()      │
                    │ + aggregate_by_group()     │
                    │ + calculate_summary_stats()│
                    │ + pivot_and_aggregate()    │
                    │ + run_pipeline()           │
                    │ + process()                │
                    └────────────────────────────┘
```

### 3. Pipeline Orchestrator Hierarchy

```
                    ┌────────────────────────────┐
                    │        BasePipeline        │ (ABC)
                    ├────────────────────────────┤
                    │ + name: str                │
                    ├────────────────────────────┤
                    │ + add_processor() [abstract│
                    │ + execute() [abstract]     │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │        DataPipeline        │
                    ├────────────────────────────┤
                    │ + loader: Optional[BaseDL] │
                    │ + processors: List[BaseDP] │
                    │ + execution_summary: List  │
                    ├────────────────────────────┤
                    │ + set_loader(loader)       │
                    │ + add_processor(processor) │
                    │ + clear_processors()       │
                    │ + execute(initial_df=None) │
                    │ + __len__(), __iter__()    │
                    └────────────────────────────┘
```

---

## Method Organization & OOP Class Contracts

### 1. `BaseDataLoader` & `DataLoader` (`data_loader.py`)

- **`BaseDataLoader` (ABC)**:
  - Enforces `load()` and `save()` implementation via `@abstractmethod`.
  - Encapsulates path resolution (`_resolve_source_path`, `_resolve_dest_path`) with validation against non-existent paths or directories.
  - Implements state properties (`data`, `is_loaded`).
  - Implements shared data cleansing (`clean()`) and structural metadata inspection (`get_summary()`).
- **`DataLoader` (CSVDataLoader)**:
  - Configurable `delimiter` and `encoding`.
  - Caches loaded DataFrames in `self._data` while supporting ad-hoc path overrides.
- **`JSONDataLoader`**:
  - Demonstrates inheritance and polymorphism.
  - Supports configurable `orient` ('records', 'split', 'index', etc.) and indentation.

### 2. `BaseDataProcessor` & Subclasses (`data_processor.py`)

- **`BaseDataProcessor` (ABC)**:
  - Enforces `process(df: pd.DataFrame) -> pd.DataFrame`.
  - Implements `__call__` so processors can be passed directly as callables.
  - Houses execution audit logging (`log_step`, `audit_log`, `reset_audit_log`).
- **`DataValidator`**:
  - Focuses strictly on validation contracts: `validate_schema` and `validate_column_ranges`.
  - `process(df)` executes preset constraints and raises descriptive exceptions on violation.
- **`DataTransformer`**:
  - Focuses on transformations: `filter_by_condition`, `normalize_numeric_columns`, `cast_column_types`.
  - Supports step configuration via `add_filter()`.
  - `process(df)` applies filters, normalization, and type casting sequentially.
- **`DataAggregator`**:
  - Focuses on aggregation: `aggregate_by_group`, `calculate_summary_statistics`, `pivot_and_aggregate`.
  - `process(df)` aggregates data if grouping parameters are configured.
- **`DataProcessor`**:
  - Comprehensive unified processor coordinating validation, transformation, and aggregation.
  - Organizes methods into clean logical domains.
  - Offers a fluent API (`set_data()`, `filter()`, `normalize()`, `cast()`, `get_data()`).

### 3. `DataPipeline` (`pipeline.py`)

- Connects an ingestion stage (`BaseDataLoader`) to any number of processing stages (`BaseDataProcessor`).
- Tracks fine-grained execution metrics: stage name, component class, input shape, output shape, and duration in seconds.
- Factory helper functions:
  - `build_pipeline(loader, processors, name)`
  - `create_standard_pipeline(file_path, required_columns, ...)`

---

## Demonstration of Success Criteria

### 1. Type Hints
- **Strict Typing Compliance**: All modules (`data_loader.py`, `data_processor.py`, `pipeline.py`, `test_oop_pipeline.py`) pass `python -m mypy --strict` with zero issues.
- **Standard Type Annotations**: Utilizes `from __future__ import annotations`, `Union`, `Optional`, `Sequence`, `Mapping`, `Tuple`, `Dict`, `List`, `Any`, `Callable`, `Iterator`, and `Path`.
- **Structural Subtyping**: Uses `Sequence[str]` and `Mapping[str, ...]` to accept lists, tuples, dictionaries, and custom mappings seamlessly without forcing concrete types.

### 2. Functions & Modules
- **Modular Boundaries**: Clear separation across data loading (`data_loader`), data manipulation (`data_processor`), and workflow orchestration (`pipeline`).
- **Explicit API Surface**: Each module defines `__all__` to explicitly expose intended classes and functions.
- **Cohesive OOP + Functional Integration**: Procedural convenience functions (`load_csv`, `transform_data`, `save_csv`, `validate_schema`, etc.) coexist alongside OOP classes, guaranteeing 100% backward compatibility for existing callers.

---

## Code Examples

### Example 1: Object-Oriented DataLoader & Ingestion

```python
from pathlib import Path
from data_loader import DataLoader, JSONDataLoader

# 1. Initialize CSV loader
csv_loader = DataLoader(file_path="data/raw_data.csv")
df = csv_loader.load()

# 2. Inspect metadata summary
summary = csv_loader.get_summary()
print("Row count:", summary["num_rows"])

# 3. Clean loaded data in-place
cleaned_df = csv_loader.clean(drop_na=True, strip_whitespace=True)

# 4. Save to JSON using polymorphic JSONDataLoader
json_loader = JSONDataLoader()
json_loader.save(df=cleaned_df, destination="data/cleaned.json", orient="records")
```

### Example 2: Fluent Method Chaining with `DataProcessor`

```python
import pandas as pd
from data_processor import DataProcessor

df = pd.DataFrame({
    "department": ["Engineering", "Marketing", "Engineering", "HR"],
    "salary": [120000.0, 95000.0, 110000.0, 85000.0],
    "experience": [5, 3, 4, 2],
})

# Fluent chaining
processor = DataProcessor()
result = (
    processor.set_data(df)
    .filter("salary", ">=", 95000.0)
    .normalize(["salary"], method="minmax")
    .cast({"experience": "int64"})
    .get_data()
)

print(result)
print("Audit Log:", processor.audit_log)
```

### Example 3: End-to-End Orchestration with `DataPipeline`

```python
from data_loader import DataLoader
from data_processor import DataValidator, DataTransformer, DataAggregator
from pipeline import DataPipeline

# 1. Assemble stages
pipeline = (
    DataPipeline(loader=DataLoader(file_path="employees.csv"))
    .add_processor(DataValidator(required_columns=["department", "salary", "experience"]))
    .add_processor(DataTransformer(filters=[("experience", ">=", 3)]))
    .add_processor(DataAggregator(group_by="department", aggregations={"salary": ["mean", "count"]}))
)

# 2. Execute pipeline
aggregated_df = pipeline.execute()

# 3. Inspect telemetry
for record in pipeline.execution_summary:
    print(f"{record['stage']}: {record['input_shape']} -> {record['output_shape']} in {record['duration_seconds']}s")
```

---

## Verification & Automated Testing

### 1. Static Type Checking with Mypy (Strict Mode)

```bash
python -m mypy --strict data_loader.py data_processor.py pipeline.py
```

Output:
```
Success: no issues found in 3 source files
```

### 2. Running Automated Test Suite with Pytest

```bash
python -m pytest -v
```

**Results:**
- **138 tests passed** across root suites and practice folders:
  - `tests/test_data_loader.py`: 12 tests (loading, cleaning, edge cases)
  - `tests/test_data_processor.py`: 30 tests (validation, transforms, aggregations)
  - `tests/test_oop_pipeline.py`: 27 tests (OOP hierarchies, inheritance, polymorphism, chaining, telemetry)
  - `practice/week-01-.../test_data_loader.py`: 12 tests
  - `practice/week-02-.../test_data_processor.py`: 30 tests
  - `practice/week-03-.../test_oop_pipeline.py`: 27 tests
