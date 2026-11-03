"""
data_processor.py

A modular, reusable Python data processing pipeline component providing
type-hinted functions for dataset validation, transformation, and aggregation
with strict type contracts and comprehensive docstrings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import pandas as pd  # type: ignore[import-untyped]

__all__ = [
    # OOP Classes
    "BaseDataProcessor",
    "DataValidator",
    "DataTransformer",
    "DataAggregator",
    "DataProcessor",
    # Functional APIs
    "validate_schema",
    "validate_column_ranges",
    "validate_null_constraints",
    "validate_unique_constraints",
    "validate_allowed_values",
    "validate_custom_rules",
    "get_null_violations",
    "get_unique_violations",
    "get_allowed_values_violations",
    "filter_by_condition",
    "normalize_numeric_columns",
    "cast_column_types",
    "aggregate_by_group",
    "calculate_summary_statistics",
    "pivot_and_aggregate",
    "run_pipeline",
]


# ============================================================================
# Section 1: Validation Functions
# ============================================================================

def validate_schema(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    column_types: Optional[Mapping[str, Union[type, str]]] = None,
    allow_empty: bool = False,
) -> bool:
    """
    Validate that a DataFrame conforms to expected column schema and data type constraints.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to validate.
    required_columns : Sequence[str]
        A sequence (e.g., list, tuple) of column names that must be present in the DataFrame.
    column_types : Optional[Mapping[str, Union[type, str]]], default None
        Optional mapping of column names to expected Python types (e.g., int, float, str, bool)
        or pandas dtype strings (e.g., 'int64', 'float64', 'object', 'datetime64[ns]').
    allow_empty : bool, default False
        Whether an empty DataFrame (0 rows) is considered valid.

    Returns
    -------
    bool
        True if the DataFrame satisfies all validation criteria.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame, or if any column in column_types has an incompatible type.
    ValueError
        If df is empty when allow_empty is False, or if any required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    if not allow_empty and df.empty:
        raise ValueError("DataFrame is empty, which violates the schema constraint (allow_empty=False).")

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

    if column_types:
        for col, expected_type in column_types.items():
            if col not in df.columns:
                raise ValueError(f"Column '{col}' specified in column_types is missing from DataFrame.")

            if isinstance(expected_type, type):
                if expected_type is int:
                    if not pd.api.types.is_integer_dtype(df[col]):
                        raise TypeError(f"Column '{col}' expected integer type, got dtype '{df[col].dtype}'.")
                elif expected_type is float:
                    if not pd.api.types.is_float_dtype(df[col]):
                        raise TypeError(f"Column '{col}' expected float type, got dtype '{df[col].dtype}'.")
                elif expected_type is str:
                    if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
                        raise TypeError(f"Column '{col}' expected string type, got dtype '{df[col].dtype}'.")
                elif expected_type is bool:
                    if not pd.api.types.is_bool_dtype(df[col]):
                        raise TypeError(f"Column '{col}' expected bool type, got dtype '{df[col].dtype}'.")
                else:
                    sample_values = df[col].dropna()
                    if not sample_values.empty and not all(isinstance(val, expected_type) for val in sample_values):
                        raise TypeError(
                            f"Column '{col}' contains elements not matching expected type '{expected_type.__name__}'."
                        )
            elif isinstance(expected_type, str):
                col_dtype_str = str(df[col].dtype).lower()
                expected_str = expected_type.lower()
                if expected_str not in col_dtype_str:
                    raise TypeError(
                        f"Column '{col}' expected dtype '{expected_type}', got '{df[col].dtype}'."
                    )

    return True


def validate_column_ranges(
    df: pd.DataFrame,
    bounds: Mapping[str, Tuple[Optional[float], Optional[float]]],
) -> Dict[str, bool]:
    """
    Validate that numeric values in specified columns fall within min and max boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to inspect.
    bounds : Mapping[str, Tuple[Optional[float], Optional[float]]]
        Mapping from column names to a (min_val, max_val) tuple.
        Use None if a boundary is unrestricted (e.g., (0.0, None) for non-negative values).

    Returns
    -------
    Dict[str, bool]
        Dictionary mapping each column to a boolean indicating whether all non-null values
        satisfy the boundary constraints.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame, or if any checked column is not numeric.
    KeyError
        If any column specified in bounds does not exist in df.
    ValueError
        If for any column min_val > max_val, or if bounds mapping is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    if not bounds:
        raise ValueError("bounds mapping cannot be empty.")

    results: Dict[str, bool] = {}

    for col, (min_val, max_val) in bounds.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' specified in bounds not found in DataFrame.")

        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' is not numeric (dtype: {df[col].dtype}).")

        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValueError(
                f"Invalid bounds for column '{col}': min_val ({min_val}) cannot exceed max_val ({max_val})."
            )

        series = df[col].dropna()
        if series.empty:
            results[col] = True
            continue

        is_valid = True
        if min_val is not None:
            is_valid = is_valid and bool((series >= min_val).all())
        if max_val is not None:
            is_valid = is_valid and bool((series <= max_val).all())

        results[col] = is_valid

    return results


def validate_null_constraints(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    max_missing_ratio: float = 0.0,
    raise_on_error: bool = False,
) -> Dict[str, bool]:
    """
    Validate that missing/null values in specified columns do not exceed allowed thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to inspect.
    columns : Optional[Sequence[str]], default None
        Sequence of column names to validate. If None, all columns in df are checked.
    max_missing_ratio : float, default 0.0
        Maximum allowable proportion of missing (null/NaN) values, between 0.0 and 1.0.
        0.0 means strictly zero missing values are allowed (complete data required).
    raise_on_error : bool, default False
        Whether to raise a ValueError if any column exceeds the allowed missing ratio.

    Returns
    -------
    Dict[str, bool]
        Dictionary mapping each validated column to a boolean indicating whether it
        satisfied the missing ratio constraint.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame, or if max_missing_ratio is not numeric.
    ValueError
        If max_missing_ratio is outside [0.0, 1.0], or if raise_on_error is True and a
        column exceeds the allowable missing ratio.
    KeyError
        If any specified column is missing from df.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    if not isinstance(max_missing_ratio, (int, float)) or isinstance(max_missing_ratio, bool):
        raise TypeError(f"Expected numeric max_missing_ratio, got {type(max_missing_ratio).__name__}")

    if not (0.0 <= max_missing_ratio <= 1.0):
        raise ValueError(
            f"max_missing_ratio must be between 0.0 and 1.0 inclusive, got {max_missing_ratio}."
        )

    cols = list(columns) if columns is not None else list(df.columns)
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' specified in columns not found in DataFrame.")

    results: Dict[str, bool] = {}
    total_rows = len(df)

    for col in cols:
        if total_rows == 0:
            results[col] = True
            continue

        missing_count = int(df[col].isna().sum())
        missing_ratio = float(missing_count / total_rows)
        results[col] = bool(missing_ratio <= max_missing_ratio)

    if raise_on_error:
        failed_cols = [c for c, passed in results.items() if not passed]
        if failed_cols:
            raise ValueError(
                f"Null constraint validation failed for columns: {failed_cols} "
                f"(max allowed missing ratio: {max_missing_ratio})"
            )

    return results


def get_null_violations(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    max_missing_ratio: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute detailed null value violation metrics for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to inspect.
    columns : Optional[Sequence[str]], default None
        Sequence of column names to evaluate. If None, all columns are evaluated.
    max_missing_ratio : float, default 0.0
        Maximum allowable missing ratio [0.0, 1.0].

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Diagnostics dictionary mapping each column to:
        - 'missing_count' (int): Total null/missing values in column.
        - 'missing_ratio' (float): Ratio of missing values to total rows.
        - 'max_allowed_ratio' (float): Threshold configured.
        - 'passed' (bool): True if missing_ratio <= max_missing_ratio.

    Raises
    ------
    TypeError
        If df is not a pd.DataFrame or max_missing_ratio is not numeric.
    ValueError
        If max_missing_ratio is outside [0.0, 1.0].
    KeyError
        If any specified column is missing from df.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    if not isinstance(max_missing_ratio, (int, float)) or isinstance(max_missing_ratio, bool):
        raise TypeError(f"Expected numeric max_missing_ratio, got {type(max_missing_ratio).__name__}")

    if not (0.0 <= max_missing_ratio <= 1.0):
        raise ValueError(
            f"max_missing_ratio must be between 0.0 and 1.0 inclusive, got {max_missing_ratio}."
        )

    cols = list(columns) if columns is not None else list(df.columns)
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' specified in columns not found in DataFrame.")

    total_rows = len(df)
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for col in cols:
        if total_rows == 0:
            diagnostics[col] = {
                "missing_count": 0,
                "missing_ratio": 0.0,
                "max_allowed_ratio": float(max_missing_ratio),
                "passed": True,
            }
            continue

        missing_count = int(df[col].isna().sum())
        missing_ratio = float(missing_count / total_rows)
        diagnostics[col] = {
            "missing_count": missing_count,
            "missing_ratio": missing_ratio,
            "max_allowed_ratio": float(max_missing_ratio),
            "passed": bool(missing_ratio <= max_missing_ratio),
        }

    return diagnostics


def validate_unique_constraints(
    df: pd.DataFrame,
    columns: Union[str, Sequence[str]],
    allow_na: bool = False,
    raise_on_error: bool = False,
) -> bool:
    """
    Validate that a column or composite combination of columns contains strictly unique values.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to inspect.
    columns : Union[str, Sequence[str]]
        A single column name or sequence of column names forming the uniqueness key.
    allow_na : bool, default False
        Whether null/NaN values are ignored during uniqueness checks.
        If False, presence of any nulls violates uniqueness.
    raise_on_error : bool, default False
        Whether to raise a ValueError if duplicate rows exist.

    Returns
    -------
    bool
        True if all rows are unique across the specified column(s), False otherwise.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame, or columns argument is not a str or Sequence[str].
    KeyError
        If any specified column is missing from df.
    ValueError
        If columns is empty, or if raise_on_error is True and duplicate rows exist.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    if isinstance(columns, str):
        cols = [columns]
    elif isinstance(columns, (list, tuple, Sequence)):
        cols = list(columns)
    else:
        raise TypeError(f"Expected columns to be str or Sequence[str], got {type(columns).__name__}")

    if not cols:
        raise ValueError("columns cannot be empty.")

    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' specified in columns not found in DataFrame.")

    if df.empty:
        return True

    subset = df[cols]
    if not allow_na and bool(subset.isna().any().any()):
        if raise_on_error:
            raise ValueError(
                f"Uniqueness constraint failed: column(s) {cols} contain null values and allow_na=False."
            )
        return False

    if allow_na:
        subset = subset.dropna()

    is_unique = not bool(subset.duplicated().any())
    if raise_on_error and not is_unique:
        duplicates_count = int(subset.duplicated().sum())
        raise ValueError(
            f"Uniqueness constraint validation failed for column(s) {cols}: "
            f"found {duplicates_count} duplicate row(s)."
        )

    return is_unique


def get_unique_violations(
    df: pd.DataFrame,
    columns: Union[str, Sequence[str]],
    allow_na: bool = False,
) -> Dict[str, Any]:
    """
    Inspect duplicate records and compile violation diagnostics for uniqueness constraints.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to inspect.
    columns : Union[str, Sequence[str]]
        A single column name or sequence of column names forming the uniqueness key.
    allow_na : bool, default False
        Whether null/NaN values are ignored during uniqueness checks.

    Returns
    -------
    Dict[str, Any]
        Diagnostics dictionary containing:
        - 'columns' (List[str]): List of column names evaluated.
        - 'is_unique' (bool): True if all records are unique across columns.
        - 'duplicate_count' (int): Total number of duplicate rows found.
        - 'sample_duplicates' (List[Any]): Up to 5 sample duplicate values or tuples.

    Raises
    ------
    TypeError
        If df is not a pd.DataFrame or columns argument is invalid.
    KeyError
        If any specified column is missing from df.
    ValueError
        If columns is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    if isinstance(columns, str):
        cols = [columns]
    elif isinstance(columns, (list, tuple, Sequence)):
        cols = list(columns)
    else:
        raise TypeError(f"Expected columns to be str or Sequence[str], got {type(columns).__name__}")

    if not cols:
        raise ValueError("columns cannot be empty.")

    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' specified in columns not found in DataFrame.")

    if df.empty:
        return {
            "columns": cols,
            "is_unique": True,
            "duplicate_count": 0,
            "sample_duplicates": [],
        }

    subset = df[cols]
    if not allow_na and bool(subset.isna().any().any()):
        null_count = int(subset.isna().any(axis=1).sum())
        return {
            "columns": cols,
            "is_unique": False,
            "duplicate_count": null_count,
            "sample_duplicates": ["<null_entry>"],
            "has_nulls": True,
        }

    if allow_na:
        subset = subset.dropna()

    dup_mask = subset.duplicated(keep=False)
    dup_rows = subset[dup_mask]
    dup_count = int(subset.duplicated().sum())

    sample_dups: List[Any] = []
    if not dup_rows.empty:
        if len(cols) == 1:
            sample_dups = list(dup_rows[cols[0]].unique()[:5])
        else:
            sample_dups = [tuple(x) for x in dup_rows.drop_duplicates().values[:5]]

    return {
        "columns": cols,
        "is_unique": dup_count == 0,
        "duplicate_count": dup_count,
        "sample_duplicates": sample_dups,
    }


def validate_allowed_values(
    df: pd.DataFrame,
    allowed_values: Mapping[str, Sequence[Any]],
    allow_null: bool = True,
    raise_on_error: bool = False,
    ignore_case: bool = False,
    strip_whitespace: bool = False,
) -> Dict[str, bool]:
    """
    Validate that values in specified columns belong to an allowed set of discrete values.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to inspect.
    allowed_values : Mapping[str, Sequence[Any]]
        Mapping of column names to valid/allowable values.
    allow_null : bool, default True
        Whether null/NaN values are considered permissible.
    raise_on_error : bool, default False
        Whether to raise a ValueError if invalid values are encountered.
    ignore_case : bool, default False
        Whether string comparisons should ignore case differences.
    strip_whitespace : bool, default False
        Whether string values should have surrounding whitespace stripped before validation.

    Returns
    -------
    Dict[str, bool]
        Dictionary mapping column names to a boolean indicating validity.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame, or allowed_values is not a Mapping.
    KeyError
        If any specified column in allowed_values is missing from df.
    ValueError
        If allowed_values is empty, or if raise_on_error is True and invalid values exist.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    if not isinstance(allowed_values, Mapping):
        raise TypeError(f"Expected allowed_values to be a Mapping, got {type(allowed_values).__name__}")

    if not allowed_values:
        raise ValueError("allowed_values mapping cannot be empty.")

    for col in allowed_values.keys():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' specified in allowed_values not found in DataFrame.")

    results: Dict[str, bool] = {}

    for col, valid_set in allowed_values.items():
        series = df[col]
        if allow_null:
            series = series.dropna()

        if series.empty:
            results[col] = True
            continue

        if ignore_case or strip_whitespace:
            def _norm(val: Any) -> Any:
                if isinstance(val, str):
                    if strip_whitespace:
                        val = val.strip()
                    if ignore_case:
                        val = val.lower()
                return val

            normalized_series = series.map(_norm)
            normalized_valid = {_norm(v) for v in valid_set}
            is_valid = bool(normalized_series.isin(normalized_valid).all())
        else:
            valid_collection = set(valid_set)
            is_valid = bool(series.isin(valid_collection).all())

        results[col] = is_valid

    if raise_on_error:
        failed = [col for col, ok in results.items() if not ok]
        if failed:
            raise ValueError(f"Allowed values constraint violated for columns: {failed}")

    return results


def get_allowed_values_violations(
    df: pd.DataFrame,
    allowed_values: Mapping[str, Sequence[Any]],
    allow_null: bool = True,
    ignore_case: bool = False,
    strip_whitespace: bool = False,
) -> Dict[str, List[Any]]:
    """
    Identify distinct unexpected values violating categorical domain constraints.

    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame to inspect.
    allowed_values : Mapping[str, Sequence[Any]]
        Mapping of column names to valid/permitted discrete values.
    allow_null : bool, default True
        Whether null/NaN values are considered permissible.
    ignore_case : bool, default False
        Whether string values are compared case-insensitively.
    strip_whitespace : bool, default False
        Whether leading/trailing whitespace is stripped before comparison.

    Returns
    -------
    Dict[str, List[Any]]
        Dictionary mapping each column to a list of distinct unexpected values observed.

    Raises
    ------
    TypeError
        If df is not a pd.DataFrame or allowed_values is not a Mapping.
    KeyError
        If any specified column in allowed_values is missing from df.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    if not isinstance(allowed_values, Mapping):
        raise TypeError(f"Expected allowed_values to be a Mapping, got {type(allowed_values).__name__}")

    violations: Dict[str, List[Any]] = {}

    for col, valid_set in allowed_values.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' specified in allowed_values not found in DataFrame.")

        series = df[col]
        if allow_null:
            series = series.dropna()

        if series.empty:
            violations[col] = []
            continue

        if ignore_case or strip_whitespace:
            def _norm(val: Any) -> Any:
                if isinstance(val, str):
                    if strip_whitespace:
                        val = val.strip()
                    if ignore_case:
                        val = val.lower()
                return val

            normalized_valid = {_norm(v) for v in valid_set}
            bad_mask = ~series.map(_norm).isin(normalized_valid)
        else:
            valid_collection = set(valid_set)
            bad_mask = ~series.isin(valid_collection)

        unexpected = list(series[bad_mask].unique())
        violations[col] = unexpected

    return violations


def validate_custom_rules(
    df: pd.DataFrame,
    rules: Mapping[str, Callable[[pd.DataFrame], Union[bool, pd.Series]]],
    raise_on_error: bool = False,
) -> Dict[str, bool]:
    """
    Validate custom user-defined validation rules against a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame to validate.
    rules : Mapping[str, Callable[[pd.DataFrame], Union[bool, pd.Series]]]
        Mapping from rule names to callables. Each callable accepts df and returns either
        a boolean or a boolean Series.
    raise_on_error : bool, default False
        Whether to raise ValueError immediately if any custom rule fails.

    Returns
    -------
    Dict[str, bool]
        Mapping from rule name to boolean indicating whether the rule passed completely.

    Raises
    ------
    TypeError
        If df is not a pd.DataFrame or rules is not a Mapping.
    ValueError
        If rules is empty, or if raise_on_error is True and any rule fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    if not isinstance(rules, Mapping):
        raise TypeError(f"Expected rules to be a Mapping, got {type(rules).__name__}")

    if not rules:
        raise ValueError("rules mapping cannot be empty.")

    results: Dict[str, bool] = {}

    for rule_name, rule_fn in rules.items():
        try:
            res = rule_fn(df)
            if isinstance(res, pd.Series):
                passed = bool(res.all())
            elif isinstance(res, bool):
                passed = res
            else:
                passed = bool(res)
        except Exception as err:
            if raise_on_error:
                raise ValueError(f"Custom rule '{rule_name}' raised error: {err}") from err
            passed = False

        results[rule_name] = passed

    if raise_on_error:
        failed_rules = [r for r, ok in results.items() if not ok]
        if failed_rules:
            raise ValueError(f"Custom validation rule(s) failed: {failed_rules}")

    return results


# ============================================================================
# Section 2: Transformation Functions
# ============================================================================

def filter_by_condition(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value: Any,
) -> pd.DataFrame:
    """
    Filter DataFrame rows based on a comparison operator and target value.

    Parameters
    ----------
    df : pd.DataFrame
        The source pandas DataFrame to filter.
    column : str
        The column name to evaluate.
    operator : str
        Comparison operator. Supported values:
        '==', '!=', '>', '>=', '<', '<=', 'in', 'not in'.
    value : Any
        The reference value or collection to compare column values against.
        Must be an iterable collection (e.g., list, set, tuple) for 'in' / 'not in'.

    Returns
    -------
    pd.DataFrame
        A new filtered DataFrame with consecutive reindexed rows.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    KeyError
        If column does not exist in df.
    ValueError
        If operator is unsupported or if value is not an iterable for 'in' / 'not in'.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    if column not in df.columns:
        raise KeyError(f"Column '{column}' does not exist in DataFrame.")

    op = operator.strip().lower()
    series = df[column]

    if op == "==":
        mask = series == value
    elif op == "!=":
        mask = series != value
    elif op == ">":
        mask = series > value
    elif op == ">=":
        mask = series >= value
    elif op == "<":
        mask = series < value
    elif op == "<=":
        mask = series <= value
    elif op == "in":
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            raise ValueError(
                f"Value for 'in' operator must be an iterable collection, got {type(value).__name__}."
            )
        mask = series.isin(list(value))
    elif op == "not in":
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            raise ValueError(
                f"Value for 'not in' operator must be an iterable collection, got {type(value).__name__}."
            )
        mask = ~series.isin(list(value))
    else:
        raise ValueError(
            f"Unsupported operator '{operator}'. Supported operators: '==', '!=', '>', '>=', '<', '<=', 'in', 'not in'."
        )

    filtered_df: pd.DataFrame = df.loc[mask].reset_index(drop=True).copy()
    return filtered_df


def normalize_numeric_columns(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    method: str = "minmax",
) -> pd.DataFrame:
    """
    Normalize numeric columns in a DataFrame using min-max scaling or z-score standardization.

    Parameters
    ----------
    df : pd.DataFrame
        The input pandas DataFrame.
    columns : Optional[Sequence[str]], default None
        Sequence of numeric column names to normalize.
        If None, all detected numeric columns will be normalized.
    method : str, default "minmax"
        Normalization technique to apply:
        - "minmax": Scales values to [0.0, 1.0].
        - "zscore": Standardizes values to mean=0.0 and std=1.0.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with normalized values for specified numeric columns.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame, or if any targeted column is not numeric.
    KeyError
        If any column in columns does not exist in df.
    ValueError
        If method is neither 'minmax' nor 'zscore'.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    norm_method = method.strip().lower()
    if norm_method not in ("minmax", "zscore"):
        raise ValueError(
            f"Unsupported normalization method: '{method}'. Choose 'minmax' or 'zscore'."
        )

    if columns is None:
        target_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    else:
        target_cols = list(columns)
        for col in target_cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' specified for normalization not found in DataFrame.")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' is not numeric (dtype: {df[col].dtype}).")

    result: pd.DataFrame = df.copy()

    for col in target_cols:
        series = result[col].astype(float)
        if norm_method == "minmax":
            min_val = float(series.min())
            max_val = float(series.max())
            spread = max_val - min_val
            if spread == 0.0 or pd.isna(spread):
                result[col] = 0.0
            else:
                result[col] = (series - min_val) / spread
        elif norm_method == "zscore":
            mean_val = float(series.mean())
            std_val = float(series.std(ddof=0))
            if std_val == 0.0 or pd.isna(std_val):
                result[col] = 0.0
            else:
                result[col] = (series - mean_val) / std_val

    return result


def cast_column_types(
    df: pd.DataFrame,
    type_map: Mapping[str, str],
) -> pd.DataFrame:
    """
    Safely cast specified DataFrame columns to target data types.

    Parameters
    ----------
    df : pd.DataFrame
        The source pandas DataFrame.
    type_map : Mapping[str, str]
        Mapping from column names to target dtype strings
        (e.g., {'id': 'int64', 'rating': 'float64', 'created_at': 'datetime64[ns]'}).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with casted column types.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    KeyError
        If any column in type_map is missing from df.
    ValueError
        If casting fails for any column due to invalid values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    result: pd.DataFrame = df.copy()

    for col, target_type in type_map.items():
        if col not in result.columns:
            raise KeyError(f"Column '{col}' specified in type_map not found in DataFrame.")

        target_lower = target_type.strip().lower()
        try:
            if target_lower in ("datetime", "datetime64", "datetime64[ns]", "timestamp"):
                result[col] = pd.to_datetime(result[col])
            else:
                result[col] = result[col].astype(target_type)
        except Exception as err:
            raise ValueError(
                f"Failed to cast column '{col}' to '{target_type}': {err}"
            ) from err

    return result


# ============================================================================
# Section 3: Aggregation Functions
# ============================================================================

def aggregate_by_group(
    df: pd.DataFrame,
    group_by: Union[str, Sequence[str]],
    aggregations: Mapping[str, Union[str, Sequence[str]]],
    reset_index: bool = True,
) -> pd.DataFrame:
    """
    Group a DataFrame by specified keys and compute aggregated metrics.

    Parameters
    ----------
    df : pd.DataFrame
        The source pandas DataFrame.
    group_by : Union[str, Sequence[str]]
        One or more column names to group by.
    aggregations : Mapping[str, Union[str, Sequence[str]]]
        Mapping from column names to aggregation function(s),
        e.g., {'sales': ['sum', 'mean'], 'units': 'sum'}.
    reset_index : bool, default True
        Whether to reset the index so grouping columns remain standard columns.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with clear, flattened column headers.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    KeyError
        If any grouping column or aggregation target column is not found in df.
    ValueError
        If group_by or aggregations is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    group_cols: List[str] = [group_by] if isinstance(group_by, str) else list(group_by)
    if not group_cols:
        raise ValueError("group_by cannot be empty.")

    if not aggregations:
        raise ValueError("aggregations dictionary cannot be empty.")

    for col in group_cols:
        if col not in df.columns:
            raise KeyError(f"Grouping column '{col}' not found in DataFrame.")

    for col in aggregations:
        if col not in df.columns:
            raise KeyError(f"Aggregation target column '{col}' not found in DataFrame.")

    # Convert mapping to dict for pandas groupby agg
    agg_dict = dict(aggregations)
    aggregated: pd.DataFrame = df.groupby(group_cols).agg(agg_dict)

    if isinstance(aggregated.columns, pd.MultiIndex):
        flattened_names: List[str] = []
        for col_name, agg_name in aggregated.columns:
            if agg_name:
                flattened_names.append(f"{col_name}_{agg_name}")
            else:
                flattened_names.append(str(col_name))
        aggregated.columns = pd.Index(flattened_names)

    if reset_index:
        aggregated = aggregated.reset_index()

    return aggregated


def calculate_summary_statistics(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive descriptive statistics for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to analyze.
    columns : Optional[Sequence[str]], default None
        Sequence of numeric column names to analyze.
        If None, all numeric columns in df are analyzed.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary where each column maps to its statistical metrics:
        'count', 'mean', 'std', 'min', 'median', 'max'.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame, or if any specified column is non-numeric.
    KeyError
        If any column in columns is missing from df.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    if columns is None:
        target_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    else:
        target_cols = list(columns)
        for col in target_cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' specified for summary statistics not found in DataFrame.")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' is not numeric (dtype: {df[col].dtype}).")

    summary: Dict[str, Dict[str, float]] = {}

    for col in target_cols:
        series = df[col].dropna()
        if series.empty:
            summary[col] = {
                "count": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "median": 0.0,
                "max": 0.0,
            }
        else:
            summary[col] = {
                "count": float(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "median": float(series.median()),
                "max": float(series.max()),
            }

    return summary


def pivot_and_aggregate(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: str = "sum",
    fill_value: Optional[float] = 0.0,
) -> pd.DataFrame:
    """
    Create a pivot table with aggregated values and clear tabular structure.

    Parameters
    ----------
    df : pd.DataFrame
        The input pandas DataFrame.
    index : str
        Column name to use as rows/index.
    columns : str
        Column name to use as target headers.
    values : str
        Column name containing values to aggregate.
    aggfunc : str, default "sum"
        Aggregation function name (e.g., 'sum', 'mean', 'count', 'min', 'max').
    fill_value : Optional[float], default 0.0
        Value to replace missing cells in the pivot table.

    Returns
    -------
    pd.DataFrame
        Pivoted DataFrame with reset index and flattened columns.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    KeyError
        If index, columns, or values column is not in df.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    for col_name, role in [(index, "index"), (columns, "columns"), (values, "values")]:
        if col_name not in df.columns:
            raise KeyError(f"Column '{col_name}' specified as {role} not found in DataFrame.")

    pivoted: pd.DataFrame = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        fill_value=fill_value,
    )

    pivoted = pivoted.reset_index()
    pivoted.columns.name = None
    return pivoted


# ============================================================================
# Section 4: Pipeline Orchestrator
# ============================================================================

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
    """
    Execute an end-to-end data processing pipeline combining validation,
    transformation, and aggregation steps.

    Parameters
    ----------
    df : pd.DataFrame
        The raw input DataFrame.
    required_columns : Sequence[str]
        Schema validation: columns required to be present.
    filter_column : Optional[str], default None
        Column to filter on.
    filter_op : Optional[str], default None
        Filter operator (e.g., '>', '==', 'in').
    filter_value : Optional[Any], default None
        Filter comparison target value.
    normalize_columns : Optional[Sequence[str]], default None
        Columns to scale/normalize.
    normalize_method : str, default "minmax"
        Method for normalization ('minmax' or 'zscore').
    group_by : Optional[Union[str, Sequence[str]]], default None
        Grouping column(s) for aggregation.
    aggregations : Optional[Mapping[str, Union[str, Sequence[str]]]], default None
        Aggregation specifications mapping column to agg functions.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame after executing validated pipeline stages.
    """
    # Step 1: Validate Schema
    validate_schema(df, required_columns=required_columns)
    processed_df = df.copy()

    # Step 2: Apply Filter (if configured)
    if filter_column is not None and filter_op is not None:
        processed_df = filter_by_condition(
            processed_df,
            column=filter_column,
            operator=filter_op,
            value=filter_value,
        )

    # Step 3: Apply Normalization (if configured)
    if normalize_columns is not None:
        processed_df = normalize_numeric_columns(
            processed_df,
            columns=normalize_columns,
            method=normalize_method,
        )

    # Step 4: Apply Aggregation (if configured)
    if group_by is not None and aggregations is not None:
        processed_df = aggregate_by_group(
            processed_df,
            group_by=group_by,
            aggregations=aggregations,
        )

    return processed_df


# ============================================================================
# Section 5: Object-Oriented Data Processor Classes (OOP Hierarchy)
# ============================================================================

class BaseDataProcessor(ABC):
    """
    Abstract Base Class establishing the contract for data processing pipeline stages.

    Subclasses must implement the abstract `process` method, ensuring consistent
    execution signatures across validation, transformation, and aggregation stages.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """
        Initialize the processor instance with an optional identifier and audit log.

        Parameters
        ----------
        name : Optional[str], default None
            Identifier name for this processor. Defaults to the class name.
        """
        self.name: str = name if name is not None else self.__class__.__name__
        self._audit_log: List[Dict[str, Any]] = []

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        """Return a defensive copy of the processor's execution audit log."""
        return list(self._audit_log)

    def log_step(self, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a pipeline action entry into the audit log.

        Parameters
        ----------
        action : str
            Name or label of the executed step.
        details : Optional[Dict[str, Any]], default None
            Metadata or parameters associated with this step.
        """
        entry: Dict[str, Any] = {
            "processor": self.name,
            "action": action,
            "details": details or {},
        }
        self._audit_log.append(entry)

    def reset_audit_log(self) -> None:
        """Clear all entries from the audit log."""
        self._audit_log.clear()

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to process an input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to process.

        Returns
        -------
        pd.DataFrame
            The processed DataFrame.
        """
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enable processor instances to be called directly as callables.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            The processed DataFrame.
        """
        return self.process(df)


class DataValidator(BaseDataProcessor):
    """
    Object-Oriented Data Validator inheriting from BaseDataProcessor.

    Specializes in schema validation, required column verification, column data type
    conformance, and numerical range boundary checks.
    """

    def __init__(
        self,
        required_columns: Optional[Sequence[str]] = None,
        column_types: Optional[Mapping[str, Union[type, str]]] = None,
        bounds: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = None,
        non_null_columns: Optional[Sequence[str]] = None,
        unique_columns: Optional[Union[str, Sequence[str]]] = None,
        allowed_values: Optional[Mapping[str, Sequence[Any]]] = None,
        custom_rules: Optional[Mapping[str, Callable[[pd.DataFrame], Union[bool, pd.Series]]]] = None,
        max_missing_ratio: float = 0.0,
        allow_empty: bool = False,
        error_mode: str = "raise",
        ignore_case: bool = False,
        strip_whitespace: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize DataValidator with optional preset validation constraints.

        Parameters
        ----------
        required_columns : Optional[Sequence[str]], default None
            Sequence of column names that must exist.
        column_types : Optional[Mapping[str, Union[type, str]]], default None
            Expected column types mapping.
        bounds : Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = None
            Numeric range bounds mapping.
        non_null_columns : Optional[Sequence[str]], default None
            Columns required to satisfy nullity / completeness constraints.
        unique_columns : Optional[Union[str, Sequence[str]]], default None
            Column or list of columns required to have unique values.
        allowed_values : Optional[Mapping[str, Sequence[Any]]], default None
            Categorical columns mapped to permitted discrete values.
        custom_rules : Optional[Mapping[str, Callable[[pd.DataFrame], Union[bool, pd.Series]]]], default None
            Mapping of custom rule names to validation predicate functions.
        max_missing_ratio : float, default 0.0
            Allowed proportion of missing values for non-null checks.
        allow_empty : bool, default False
            Whether empty DataFrames are allowed.
        error_mode : str, default 'raise'
            Strategy when validation violations occur: 'raise', 'warn', or 'quarantine'.
        ignore_case : bool, default False
            Whether allowed values validation ignores case for strings.
        strip_whitespace : bool, default False
            Whether allowed values validation strips whitespace for strings.
        name : Optional[str], default None
            Custom identifier name.
        """
        super().__init__(name=name)
        self.required_columns: Optional[Sequence[str]] = required_columns
        self.column_types: Optional[Mapping[str, Union[type, str]]] = column_types
        self.bounds: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = bounds
        self.non_null_columns: Optional[Sequence[str]] = non_null_columns
        self.unique_columns: Optional[Union[str, Sequence[str]]] = unique_columns
        self.allowed_values: Optional[Mapping[str, Sequence[Any]]] = allowed_values
        self.custom_rules: Optional[Mapping[str, Callable[[pd.DataFrame], Union[bool, pd.Series]]]] = custom_rules
        self.max_missing_ratio: float = max_missing_ratio
        self.allow_empty: bool = allow_empty
        self.error_mode: str = error_mode.lower() if isinstance(error_mode, str) else "raise"
        if self.error_mode not in ("raise", "warn", "quarantine"):
            raise ValueError(f"Invalid error_mode '{error_mode}'. Supported modes: 'raise', 'warn', 'quarantine'.")
        self.ignore_case: bool = ignore_case
        self.strip_whitespace: bool = strip_whitespace
        self.quarantined_data: Optional[pd.DataFrame] = None

    def validate_schema(
        self,
        df: pd.DataFrame,
        required_columns: Optional[Sequence[str]] = None,
        column_types: Optional[Mapping[str, Union[type, str]]] = None,
        allow_empty: Optional[bool] = None,
    ) -> bool:
        """
        Validate schema constraints on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Target DataFrame.
        required_columns : Optional[Sequence[str]], default None
            Required columns, defaults to preset self.required_columns.
        column_types : Optional[Mapping[str, Union[type, str]]], default None
            Column types, defaults to preset self.column_types.
        allow_empty : Optional[bool], default None
            Empty flag, defaults to self.allow_empty.

        Returns
        -------
        bool
            True if valid; raises TypeError/ValueError otherwise.
        """
        cols = required_columns if required_columns is not None else (self.required_columns or [])
        types = column_types if column_types is not None else self.column_types
        empty_ok = allow_empty if allow_empty is not None else self.allow_empty

        result = validate_schema(df, required_columns=cols, column_types=types, allow_empty=empty_ok)
        self.log_step("validate_schema", {"required_columns": list(cols), "passed": result})
        return result

    def validate_column_ranges(
        self,
        df: pd.DataFrame,
        bounds: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> Dict[str, bool]:
        """
        Validate numerical boundary ranges on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Target DataFrame.
        bounds : Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]], default None
            Column boundary specs, defaults to preset self.bounds.

        Returns
        -------
        Dict[str, bool]
            Per-column validity dictionary.
        """
        target_bounds = bounds if bounds is not None else self.bounds
        if target_bounds is None:
            raise ValueError("No numeric bounds provided or configured on DataValidator.")

        results = validate_column_ranges(df, bounds=target_bounds)
        self.log_step("validate_column_ranges", {"results": results})
        return results

    def validate_null_constraints(
        self,
        df: pd.DataFrame,
        columns: Optional[Sequence[str]] = None,
        max_missing_ratio: Optional[float] = None,
        raise_on_error: bool = False,
    ) -> Dict[str, bool]:
        """
        Validate null constraints on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Target DataFrame.
        columns : Optional[Sequence[str]], default None
            Columns to validate, defaults to preset self.non_null_columns.
        max_missing_ratio : Optional[float], default None
            Allowable missing ratio, defaults to preset self.max_missing_ratio.
        raise_on_error : bool, default False
            Whether to raise on violation.

        Returns
        -------
        Dict[str, bool]
            Per-column validity mapping.
        """
        target_cols = columns if columns is not None else self.non_null_columns
        ratio = max_missing_ratio if max_missing_ratio is not None else self.max_missing_ratio
        results = validate_null_constraints(
            df,
            columns=target_cols,
            max_missing_ratio=ratio,
            raise_on_error=raise_on_error,
        )
        self.log_step("validate_null_constraints", {"results": results})
        return results

    def validate_unique_constraints(
        self,
        df: pd.DataFrame,
        columns: Optional[Union[str, Sequence[str]]] = None,
        allow_na: bool = False,
        raise_on_error: bool = False,
    ) -> bool:
        """
        Validate uniqueness constraints on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Target DataFrame.
        columns : Optional[Union[str, Sequence[str]]], default None
            Target column(s), defaults to preset self.unique_columns.
        allow_na : bool, default False
            Whether to allow null values.
        raise_on_error : bool, default False
            Whether to raise on violation.

        Returns
        -------
        bool
            True if all rows are unique across columns.
        """
        target_cols = columns if columns is not None else self.unique_columns
        if target_cols is None:
            raise ValueError("No uniqueness columns provided or configured on DataValidator.")
        result = validate_unique_constraints(
            df,
            columns=target_cols,
            allow_na=allow_na,
            raise_on_error=raise_on_error,
        )
        self.log_step("validate_unique_constraints", {"columns": list(target_cols) if isinstance(target_cols, (list, tuple, Sequence)) and not isinstance(target_cols, str) else [target_cols], "passed": result})
        return result

    def validate_allowed_values(
        self,
        df: pd.DataFrame,
        allowed_values: Optional[Mapping[str, Sequence[Any]]] = None,
        allow_null: bool = True,
        raise_on_error: bool = False,
        ignore_case: Optional[bool] = None,
        strip_whitespace: Optional[bool] = None,
    ) -> Dict[str, bool]:
        """
        Validate discrete allowed values on categorical columns.

        Parameters
        ----------
        df : pd.DataFrame
            Target DataFrame.
        allowed_values : Optional[Mapping[str, Sequence[Any]]], default None
            Mapping of column names to valid values, defaults to self.allowed_values.
        allow_null : bool, default True
            Whether null values are permissible.
        raise_on_error : bool, default False
            Whether to raise on violation.
        ignore_case : Optional[bool], default None
            Whether to ignore case differences, defaults to self.ignore_case.
        strip_whitespace : Optional[bool], default None
            Whether to strip string whitespace, defaults to self.strip_whitespace.

        Returns
        -------
        Dict[str, bool]
            Per-column validity mapping.
        """
        target_allowed = allowed_values if allowed_values is not None else self.allowed_values
        if target_allowed is None:
            raise ValueError("No allowed values mapping provided or configured on DataValidator.")
        ic = ignore_case if ignore_case is not None else self.ignore_case
        sw = strip_whitespace if strip_whitespace is not None else self.strip_whitespace
        results = validate_allowed_values(
            df,
            allowed_values=target_allowed,
            allow_null=allow_null,
            raise_on_error=raise_on_error,
            ignore_case=ic,
            strip_whitespace=sw,
        )
        self.log_step("validate_allowed_values", {"results": results})
        return results

    def validate_custom_rules(
        self,
        df: pd.DataFrame,
        rules: Optional[Mapping[str, Callable[[pd.DataFrame], Union[bool, pd.Series]]]] = None,
        raise_on_error: bool = False,
    ) -> Dict[str, bool]:
        """Validate custom predicate rules on a DataFrame."""
        target_rules = rules if rules is not None else self.custom_rules
        if target_rules is None:
            raise ValueError("No custom rules provided or configured on DataValidator.")
        results = validate_custom_rules(df, rules=target_rules, raise_on_error=raise_on_error)
        self.log_step("validate_custom_rules", {"results": results})
        return results

    def get_allowed_values_violations(
        self,
        df: pd.DataFrame,
        allowed_values: Optional[Mapping[str, Sequence[Any]]] = None,
        allow_null: bool = True,
        ignore_case: Optional[bool] = None,
        strip_whitespace: Optional[bool] = None,
    ) -> Dict[str, List[Any]]:
        """Identify unexpected values violating allowed values constraint."""
        target_allowed = allowed_values if allowed_values is not None else self.allowed_values
        if target_allowed is None:
            raise ValueError("No allowed values mapping provided or configured on DataValidator.")
        ic = ignore_case if ignore_case is not None else self.ignore_case
        sw = strip_whitespace if strip_whitespace is not None else self.strip_whitespace
        return get_allowed_values_violations(
            df,
            allowed_values=target_allowed,
            allow_null=allow_null,
            ignore_case=ic,
            strip_whitespace=sw,
        )

    def get_unique_violations(
        self,
        df: pd.DataFrame,
        columns: Optional[Union[str, Sequence[str]]] = None,
        allow_na: bool = False,
    ) -> Dict[str, Any]:
        """Compile violation diagnostics for uniqueness constraints."""
        target_cols = columns if columns is not None else self.unique_columns
        if target_cols is None:
            raise ValueError("No uniqueness columns provided or configured on DataValidator.")
        return get_unique_violations(df, columns=target_cols, allow_na=allow_na)

    def get_null_violations(
        self,
        df: pd.DataFrame,
        columns: Optional[Sequence[str]] = None,
        max_missing_ratio: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compile detailed null value violation metrics."""
        target_cols = columns if columns is not None else self.non_null_columns
        ratio = max_missing_ratio if max_missing_ratio is not None else self.max_missing_ratio
        return get_null_violations(df, columns=target_cols, max_missing_ratio=ratio)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute configured validation constraints.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.

        Returns
        -------
        pd.DataFrame
            A defensive copy of the DataFrame (or cleaned subset in quarantine mode).

        Raises
        ------
        ValueError
            If constraints fail in 'raise' error_mode.
        TypeError
            If input is not a DataFrame or column types mismatch.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        if self.error_mode == "warn":
            if self.required_columns is not None:
                try:
                    self.validate_schema(
                        df,
                        required_columns=self.required_columns,
                        column_types=self.column_types,
                        allow_empty=self.allow_empty,
                    )
                except Exception as err:
                    self.log_step("warning_schema", {"error": str(err)})

            if self.bounds is not None:
                try:
                    r_res = self.validate_column_ranges(df, bounds=self.bounds)
                    bad = [c for c, ok in r_res.items() if not ok]
                    if bad:
                        self.log_step("warning_range_bounds", {"failed_columns": bad})
                except Exception as err:
                    self.log_step("warning_range_bounds", {"error": str(err)})

            if self.non_null_columns is not None:
                try:
                    n_res = self.validate_null_constraints(
                        df,
                        columns=self.non_null_columns,
                        max_missing_ratio=self.max_missing_ratio,
                        raise_on_error=False,
                    )
                    bad = [c for c, ok in n_res.items() if not ok]
                    if bad:
                        self.log_step("warning_null_constraints", {"failed_columns": bad})
                except Exception as err:
                    self.log_step("warning_null_constraints", {"error": str(err)})

            if self.unique_columns is not None:
                try:
                    u_ok = self.validate_unique_constraints(
                        df,
                        columns=self.unique_columns,
                        raise_on_error=False,
                    )
                    if not u_ok:
                        self.log_step("warning_unique_constraints", {"columns": self.unique_columns})
                except Exception as err:
                    self.log_step("warning_unique_constraints", {"error": str(err)})

            if self.allowed_values is not None:
                try:
                    a_res = self.validate_allowed_values(
                        df,
                        allowed_values=self.allowed_values,
                        raise_on_error=False,
                    )
                    bad = [c for c, ok in a_res.items() if not ok]
                    if bad:
                        self.log_step("warning_allowed_values", {"failed_columns": bad})
                except Exception as err:
                    self.log_step("warning_allowed_values", {"error": str(err)})

            if self.custom_rules is not None:
                try:
                    c_res = self.validate_custom_rules(
                        df,
                        rules=self.custom_rules,
                        raise_on_error=False,
                    )
                    bad = [r for r, ok in c_res.items() if not ok]
                    if bad:
                        self.log_step("warning_custom_rules", {"failed_rules": bad})
                except Exception as err:
                    self.log_step("warning_custom_rules", {"error": str(err)})

            self.log_step("process_validation_warn_mode", {"rows_validated": len(df)})
            return df.copy()

        if self.error_mode == "quarantine":
            bad_indices: set[Any] = set()

            if self.non_null_columns is not None:
                for col in self.non_null_columns:
                    if col in df.columns:
                        bad_indices.update(df[df[col].isna()].index)

            if self.unique_columns is not None:
                u_cols = [self.unique_columns] if isinstance(self.unique_columns, str) else list(self.unique_columns)
                if all(c in df.columns for c in u_cols):
                    bad_indices.update(df[df.duplicated(subset=u_cols, keep="first")].index)

            if self.bounds is not None:
                for col, (min_v, max_v) in self.bounds.items():
                    if col in df.columns:
                        s = pd.to_numeric(df[col], errors="coerce")
                        if min_v is not None:
                            bad_indices.update(df[s < min_v].index)
                        if max_v is not None:
                            bad_indices.update(df[s > max_v].index)

            if self.allowed_values is not None:
                for col, valid_set in self.allowed_values.items():
                    if col in df.columns:
                        series = df[col]
                        if self.ignore_case or self.strip_whitespace:
                            def _n(val: Any) -> Any:
                                if isinstance(val, str):
                                    if self.strip_whitespace:
                                        val = val.strip()
                                    if self.ignore_case:
                                        val = val.lower()
                                return val
                            norm_valid = {_n(v) for v in valid_set}
                            bad_indices.update(df[~series.dropna().map(_n).isin(norm_valid)].index)
                        else:
                            bad_indices.update(df[~series.dropna().isin(set(valid_set))].index)

            if self.custom_rules is not None:
                for rule_name, rule_fn in self.custom_rules.items():
                    res = rule_fn(df)
                    if isinstance(res, pd.Series):
                        bad_indices.update(df[~res].index)

            bad_list = sorted(list(bad_indices))
            quarantine_df = df.loc[bad_list].copy() if bad_list else pd.DataFrame(columns=df.columns)
            clean_df = df.drop(index=bad_list).copy() if bad_list else df.copy()
            self.quarantined_data = quarantine_df
            self.log_step("quarantine_validation", {
                "total_rows": len(df),
                "quarantined_rows": len(quarantine_df),
                "clean_rows": len(clean_df),
            })
            return clean_df

        # Default error_mode == "raise"
        if self.required_columns is not None:
            self.validate_schema(
                df,
                required_columns=self.required_columns,
                column_types=self.column_types,
                allow_empty=self.allow_empty,
            )

        if self.bounds is not None:
            range_results = self.validate_column_ranges(df, bounds=self.bounds)
            failed = [col for col, passed in range_results.items() if not passed]
            if failed:
                raise ValueError(f"Numeric range validation failed for columns: {failed}")

        if self.non_null_columns is not None:
            self.validate_null_constraints(
                df,
                columns=self.non_null_columns,
                max_missing_ratio=self.max_missing_ratio,
                raise_on_error=True,
            )

        if self.unique_columns is not None:
            self.validate_unique_constraints(
                df,
                columns=self.unique_columns,
                raise_on_error=True,
            )

        if self.allowed_values is not None:
            self.validate_allowed_values(
                df,
                allowed_values=self.allowed_values,
                raise_on_error=True,
            )

        if self.custom_rules is not None:
            self.validate_custom_rules(
                df,
                rules=self.custom_rules,
                raise_on_error=True,
            )

        self.log_step("process_validation", {"rows_validated": len(df)})
        return df.copy()


class DataTransformer(BaseDataProcessor):
    """
    Object-Oriented Data Transformer inheriting from BaseDataProcessor.

    Specializes in conditional row filtering, numerical scaling, and column type coercion.
    """

    def __init__(
        self,
        filters: Optional[Sequence[Tuple[str, str, Any]]] = None,
        normalize_cols: Optional[Sequence[str]] = None,
        norm_method: str = "minmax",
        cast_types: Optional[Mapping[str, str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize DataTransformer with optional transformation configurations.

        Parameters
        ----------
        filters : Optional[Sequence[Tuple[str, str, Any]]], default None
            Sequence of (column, operator, value) filter tuples to apply sequentially.
        normalize_cols : Optional[Sequence[str]], default None
            Columns to normalize.
        norm_method : str, default "minmax"
            Normalization algorithm ('minmax' or 'zscore').
        cast_types : Optional[Mapping[str, str]], default None
            Mapping of column names to target pandas dtype strings.
        name : Optional[str], default None
            Custom identifier name.
        """
        super().__init__(name=name)
        self.filters: List[Tuple[str, str, Any]] = list(filters) if filters else []
        self.normalize_cols: Optional[Sequence[str]] = normalize_cols
        self.norm_method: str = norm_method
        self.cast_types: Optional[Mapping[str, str]] = cast_types

    def add_filter(self, column: str, operator: str, value: Any) -> DataTransformer:
        """
        Add a filter condition tuple to the transformer pipeline.

        Parameters
        ----------
        column : str
            Target column.
        operator : str
            Comparison operator.
        value : Any
            Filter target value.

        Returns
        -------
        DataTransformer
            self for method chaining.
        """
        self.filters.append((column, operator, value))
        return self

    def filter_by_condition(
        self,
        df: pd.DataFrame,
        column: str,
        operator: str,
        value: Any,
    ) -> pd.DataFrame:
        """Apply conditional row filtering to a DataFrame."""
        result = filter_by_condition(df, column=column, operator=operator, value=value)
        self.log_step("filter_by_condition", {"column": column, "operator": operator, "rows_out": len(result)})
        return result

    def normalize_numeric_columns(
        self,
        df: pd.DataFrame,
        columns: Optional[Sequence[str]] = None,
        method: str = "minmax",
    ) -> pd.DataFrame:
        """Apply numeric column normalization to a DataFrame."""
        result = normalize_numeric_columns(df, columns=columns, method=method)
        self.log_step("normalize_numeric_columns", {"columns": list(columns) if columns else None, "method": method})
        return result

    def cast_column_types(
        self,
        df: pd.DataFrame,
        type_map: Mapping[str, str],
    ) -> pd.DataFrame:
        """Apply type casting to DataFrame columns."""
        result = cast_column_types(df, type_map=type_map)
        self.log_step("cast_column_types", {"type_map": dict(type_map)})
        return result

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute configured transformation steps in order:
        1. Sequential filtering conditions
        2. Numeric column normalization
        3. Column type casting
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        transformed = df.copy()

        for col, op, val in self.filters:
            transformed = self.filter_by_condition(transformed, column=col, operator=op, value=val)

        if self.normalize_cols is not None:
            transformed = self.normalize_numeric_columns(
                transformed,
                columns=self.normalize_cols,
                method=self.norm_method,
            )

        if self.cast_types is not None:
            transformed = self.cast_column_types(transformed, type_map=self.cast_types)

        self.log_step("process_transformation", {"rows_out": len(transformed)})
        return transformed


class DataAggregator(BaseDataProcessor):
    """
    Object-Oriented Data Aggregator inheriting from BaseDataProcessor.

    Specializes in grouping, multi-metric aggregations, descriptive statistics,
    and two-dimensional pivot table generation.
    """

    def __init__(
        self,
        group_by: Optional[Union[str, Sequence[str]]] = None,
        aggregations: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        reset_index: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize DataAggregator with optional grouping and aggregation settings.

        Parameters
        ----------
        group_by : Optional[Union[str, Sequence[str]]], default None
            Grouping column(s).
        aggregations : Optional[Mapping[str, Union[str, Sequence[str]]]], default None
            Column-to-function aggregation mapping.
        reset_index : bool, default True
            Whether to reset index and flatten column headers.
        name : Optional[str], default None
            Custom identifier name.
        """
        super().__init__(name=name)
        self.group_by: Optional[Union[str, Sequence[str]]] = group_by
        self.aggregations: Optional[Mapping[str, Union[str, Sequence[str]]]] = aggregations
        self.reset_index: bool = reset_index

    def aggregate_by_group(
        self,
        df: pd.DataFrame,
        group_by: Optional[Union[str, Sequence[str]]] = None,
        aggregations: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        reset_index: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Aggregate DataFrame across groups."""
        grp = group_by if group_by is not None else self.group_by
        aggs = aggregations if aggregations is not None else self.aggregations
        rst = reset_index if reset_index is not None else self.reset_index

        if grp is None or aggs is None:
            raise ValueError("Both 'group_by' and 'aggregations' must be provided or configured.")

        result = aggregate_by_group(df, group_by=grp, aggregations=aggs, reset_index=rst)
        self.log_step(
            "aggregate_by_group",
            {"group_by": grp if isinstance(grp, str) else list(grp), "rows_out": len(result)},
        )
        return result

    def calculate_summary_statistics(
        self,
        df: pd.DataFrame,
        columns: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistical summary dictionary for numeric columns."""
        result = calculate_summary_statistics(df, columns=columns)
        self.log_step("calculate_summary_statistics", {"columns": list(columns) if columns else None})
        return result

    def pivot_and_aggregate(
        self,
        df: pd.DataFrame,
        index: str,
        columns: str,
        values: str,
        aggfunc: str = "sum",
        fill_value: Optional[float] = 0.0,
    ) -> pd.DataFrame:
        """Construct an aggregated pivot table."""
        result = pivot_and_aggregate(
            df,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=fill_value,
        )
        self.log_step("pivot_and_aggregate", {"index": index, "columns": columns, "values": values})
        return result

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute configured aggregation if group_by and aggregations are configured;
        otherwise returns a defensive copy of df.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        if self.group_by is not None and self.aggregations is not None:
            return self.aggregate_by_group(
                df,
                group_by=self.group_by,
                aggregations=self.aggregations,
                reset_index=self.reset_index,
            )
        self.log_step("process_passthrough", {"rows": len(df)})
        return df.copy()


class DataProcessor(BaseDataProcessor):
    """
    Comprehensive Object-Oriented Data Processor inheriting from BaseDataProcessor.

    Unifies validation, transformation, and aggregation operations with support
    for stateful fluent method chaining, composition, and pipeline execution.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize DataProcessor with an optional starting DataFrame.

        Parameters
        ----------
        df : Optional[pd.DataFrame], default None
            Optional DataFrame to bind into internal state for method chaining.
        name : Optional[str], default None
            Custom identifier name.
        """
        super().__init__(name=name)
        self._df: Optional[pd.DataFrame] = df.copy() if df is not None else None
        self.validator: DataValidator = DataValidator(name=f"{self.name}.validator")
        self.transformer: DataTransformer = DataTransformer(name=f"{self.name}.transformer")
        self.aggregator: DataAggregator = DataAggregator(name=f"{self.name}.aggregator")

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Return a copy of the currently bound DataFrame, or None if not set."""
        return self._df.copy() if self._df is not None else None

    @property
    def has_data(self) -> bool:
        """Return True if a DataFrame is currently loaded in internal state."""
        return self._df is not None

    def set_data(self, df: pd.DataFrame) -> DataProcessor:
        """
        Bind a new DataFrame into processor state (fluent method).

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to bind.

        Returns
        -------
        DataProcessor
            self for method chaining.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
        self._df = df.copy()
        self.log_step("set_data", {"shape": list(df.shape)})
        return self

    def get_data(self) -> pd.DataFrame:
        """
        Retrieve a defensive copy of the current internal DataFrame.

        Returns
        -------
        pd.DataFrame
            Copy of current DataFrame.

        Raises
        ------
        ValueError
            If no DataFrame is currently stored.
        """
        if self._df is None:
            raise ValueError("DataProcessor contains no DataFrame. Call set_data() first.")
        return self._df.copy()

    # Fluent Transformation Chaining Methods
    def filter(self, column: str, operator: str, value: Any) -> DataProcessor:
        """
        Apply filter_by_condition to internal DataFrame and update state.

        Returns
        -------
        DataProcessor
            self for method chaining.
        """
        current_df = self.get_data()
        self._df = self.transformer.filter_by_condition(current_df, column=column, operator=operator, value=value)
        self.log_step("filter", {"column": column, "operator": operator, "remaining_rows": len(self._df)})
        return self

    def normalize(
        self,
        columns: Optional[Sequence[str]] = None,
        method: str = "minmax",
    ) -> DataProcessor:
        """
        Apply normalize_numeric_columns to internal DataFrame and update state.

        Returns
        -------
        DataProcessor
            self for method chaining.
        """
        current_df = self.get_data()
        self._df = self.transformer.normalize_numeric_columns(current_df, columns=columns, method=method)
        self.log_step("normalize", {"columns": list(columns) if columns else None, "method": method})
        return self

    def cast(self, type_map: Mapping[str, str]) -> DataProcessor:
        """
        Apply cast_column_types to internal DataFrame and update state.

        Returns
        -------
        DataProcessor
            self for method chaining.
        """
        current_df = self.get_data()
        self._df = self.transformer.cast_column_types(current_df, type_map=type_map)
        self.log_step("cast", {"type_map": dict(type_map)})
        return self

    # Validation Method Group
    def validate_schema(
        self,
        df: Optional[pd.DataFrame] = None,
        required_columns: Sequence[str] = (),
        column_types: Optional[Mapping[str, Union[type, str]]] = None,
        allow_empty: bool = False,
    ) -> bool:
        """Validate schema on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        result = self.validator.validate_schema(
            target_df,
            required_columns=required_columns,
            column_types=column_types,
            allow_empty=allow_empty,
        )
        self.log_step("validate_schema", {"passed": result})
        return result

    def validate_column_ranges(
        self,
        bounds: Mapping[str, Tuple[Optional[float], Optional[float]]],
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, bool]:
        """Validate numeric ranges on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        result = self.validator.validate_column_ranges(target_df, bounds=bounds)
        self.log_step("validate_column_ranges", {"results": result})
        return result

    def validate_null_constraints(
        self,
        df: Optional[pd.DataFrame] = None,
        columns: Optional[Sequence[str]] = None,
        max_missing_ratio: float = 0.0,
        raise_on_error: bool = False,
    ) -> Dict[str, bool]:
        """Validate null constraints on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        result = self.validator.validate_null_constraints(
            target_df,
            columns=columns,
            max_missing_ratio=max_missing_ratio,
            raise_on_error=raise_on_error,
        )
        self.log_step("validate_null_constraints", {"results": result})
        return result

    def validate_unique_constraints(
        self,
        columns: Union[str, Sequence[str]],
        df: Optional[pd.DataFrame] = None,
        allow_na: bool = False,
        raise_on_error: bool = False,
    ) -> bool:
        """Validate uniqueness constraints on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        result = self.validator.validate_unique_constraints(
            target_df,
            columns=columns,
            allow_na=allow_na,
            raise_on_error=raise_on_error,
        )
        self.log_step("validate_unique_constraints", {"passed": result})
        return result

    def validate_allowed_values(
        self,
        allowed_values: Mapping[str, Sequence[Any]],
        df: Optional[pd.DataFrame] = None,
        allow_null: bool = True,
        raise_on_error: bool = False,
        ignore_case: bool = False,
        strip_whitespace: bool = False,
    ) -> Dict[str, bool]:
        """Validate categorical allowed values on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        result = self.validator.validate_allowed_values(
            target_df,
            allowed_values=allowed_values,
            allow_null=allow_null,
            raise_on_error=raise_on_error,
            ignore_case=ignore_case,
            strip_whitespace=strip_whitespace,
        )
        self.log_step("validate_allowed_values", {"results": result})
        return result

    def validate_custom_rules(
        self,
        rules: Mapping[str, Callable[[pd.DataFrame], Union[bool, pd.Series]]],
        df: Optional[pd.DataFrame] = None,
        raise_on_error: bool = False,
    ) -> Dict[str, bool]:
        """Validate custom rules on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        result = self.validator.validate_custom_rules(
            target_df,
            rules=rules,
            raise_on_error=raise_on_error,
        )
        self.log_step("validate_custom_rules", {"results": result})
        return result

    def get_allowed_values_violations(
        self,
        allowed_values: Mapping[str, Sequence[Any]],
        df: Optional[pd.DataFrame] = None,
        allow_null: bool = True,
        ignore_case: bool = False,
        strip_whitespace: bool = False,
    ) -> Dict[str, List[Any]]:
        """Identify unexpected categorical values on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        return self.validator.get_allowed_values_violations(
            target_df,
            allowed_values=allowed_values,
            allow_null=allow_null,
            ignore_case=ignore_case,
            strip_whitespace=strip_whitespace,
        )

    def get_unique_violations(
        self,
        columns: Union[str, Sequence[str]],
        df: Optional[pd.DataFrame] = None,
        allow_na: bool = False,
    ) -> Dict[str, Any]:
        """Compile uniqueness violation diagnostics on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        return self.validator.get_unique_violations(
            target_df,
            columns=columns,
            allow_na=allow_na,
        )

    def get_null_violations(
        self,
        df: Optional[pd.DataFrame] = None,
        columns: Optional[Sequence[str]] = None,
        max_missing_ratio: float = 0.0,
    ) -> Dict[str, Dict[str, Any]]:
        """Compile null violation diagnostics on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        return self.validator.get_null_violations(
            target_df,
            columns=columns,
            max_missing_ratio=max_missing_ratio,
        )

    # Transformation Method Group
    def filter_by_condition(
        self,
        df: Optional[pd.DataFrame] = None,
        column: str = "",
        operator: str = "==",
        value: Any = None,
    ) -> pd.DataFrame:
        """Filter rows conditionally on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        return self.transformer.filter_by_condition(target_df, column=column, operator=operator, value=value)

    def normalize_numeric_columns(
        self,
        df: Optional[pd.DataFrame] = None,
        columns: Optional[Sequence[str]] = None,
        method: str = "minmax",
    ) -> pd.DataFrame:
        """Normalize numeric columns on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        return self.transformer.normalize_numeric_columns(target_df, columns=columns, method=method)

    def cast_column_types(
        self,
        df: Optional[pd.DataFrame] = None,
        type_map: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        """Cast column types on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        if type_map is None:
            raise ValueError("type_map must be provided.")
        return self.transformer.cast_column_types(target_df, type_map=type_map)

    # Aggregation Method Group
    def aggregate_by_group(
        self,
        df: Optional[pd.DataFrame] = None,
        group_by: Union[str, Sequence[str]] = "",
        aggregations: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        reset_index: bool = True,
    ) -> pd.DataFrame:
        """Aggregate grouped rows on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        if aggregations is None:
            raise ValueError("aggregations mapping must be provided.")
        return self.aggregator.aggregate_by_group(
            target_df,
            group_by=group_by,
            aggregations=aggregations,
            reset_index=reset_index,
        )

    def calculate_summary_statistics(
        self,
        df: Optional[pd.DataFrame] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        return self.aggregator.calculate_summary_statistics(target_df, columns=columns)

    def pivot_and_aggregate(
        self,
        df: Optional[pd.DataFrame] = None,
        index: str = "",
        columns: str = "",
        values: str = "",
        aggfunc: str = "sum",
        fill_value: Optional[float] = 0.0,
    ) -> pd.DataFrame:
        """Pivot and aggregate on provided DataFrame or internal state."""
        target_df = df if df is not None else self.get_data()
        return self.aggregator.pivot_and_aggregate(
            target_df,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=fill_value,
        )

    # Pipeline Orchestration Method
    def run_pipeline(
        self,
        df: Optional[pd.DataFrame] = None,
        required_columns: Sequence[str] = (),
        filter_column: Optional[str] = None,
        filter_op: Optional[str] = None,
        filter_value: Optional[Any] = None,
        normalize_columns: Optional[Sequence[str]] = None,
        normalize_method: str = "minmax",
        group_by: Optional[Union[str, Sequence[str]]] = None,
        aggregations: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
    ) -> pd.DataFrame:
        """
        Execute end-to-end processing pipeline on provided DataFrame or internal state.
        """
        target_df = df if df is not None else self.get_data()
        result = run_pipeline(
            target_df,
            required_columns=required_columns,
            filter_column=filter_column,
            filter_op=filter_op,
            filter_value=filter_value,
            normalize_columns=normalize_columns,
            normalize_method=normalize_method,
            group_by=group_by,
            aggregations=aggregations,
        )
        self.log_step("run_pipeline", {"rows_out": len(result)})
        return result

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standard implementation of BaseDataProcessor.process contract.
        Processes the input DataFrame and records execution.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
        self.log_step("process", {"input_rows": len(df)})
        return df.copy()


if __name__ == "__main__":
    print("data_processor module loaded successfully.")
