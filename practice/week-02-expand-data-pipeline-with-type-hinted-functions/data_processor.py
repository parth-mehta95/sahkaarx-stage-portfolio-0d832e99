"""
data_processor.py

A modular, reusable Python data processing pipeline component providing
type-hinted functions for dataset validation, transformation, and aggregation
with strict type contracts and comprehensive docstrings.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import pandas as pd  # type: ignore[import-untyped]


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


if __name__ == "__main__":
    print("data_processor module loaded successfully.")
