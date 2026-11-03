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
        allow_empty: bool = False,
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
        bounds : Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]], default None
            Numeric range bounds mapping.
        allow_empty : bool, default False
            Whether empty DataFrames are allowed.
        name : Optional[str], default None
            Custom identifier name.
        """
        super().__init__(name=name)
        self.required_columns: Optional[Sequence[str]] = required_columns
        self.column_types: Optional[Mapping[str, Union[type, str]]] = column_types
        self.bounds: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = bounds
        self.allow_empty: bool = allow_empty

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
            A defensive copy of the DataFrame if validation succeeds.

        Raises
        ------
        ValueError
            If required columns are missing, DataFrame is empty, or ranges fail.
        TypeError
            If input is not a DataFrame or column types mismatch.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

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
