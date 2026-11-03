"""
pipeline.py

An Object-Oriented pipeline orchestration module coordinating BaseDataLoader
and BaseDataProcessor stages into composable, end-to-end data workflows
with rigorous type hinting and step-level execution telemetry.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import pandas as pd  # type: ignore[import-untyped]

from data_loader import BaseDataLoader, DataLoader
from data_processor import (
    BaseDataProcessor,
    DataAggregator,
    DataProcessor,
    DataTransformer,
    DataValidator,
    validate_allowed_values,
    validate_column_ranges,
    validate_null_constraints,
    validate_schema,
    validate_unique_constraints,
)

__all__ = [
    "BasePipeline",
    "DataPipeline",
    "build_pipeline",
    "create_standard_pipeline",
    "validate_pipeline_data",
    "validate_null_constraints",
    "validate_unique_constraints",
]


class BasePipeline(ABC):
    """
    Abstract Base Class establishing the contract for pipeline orchestrators.

    Guarantees consistent stage registration and execution workflows across
    different pipeline implementations.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """
        Initialize the base pipeline orchestrator.

        Parameters
        ----------
        name : Optional[str], default None
            Identifier name for this pipeline. Defaults to class name.
        """
        self.name: str = name if name is not None else self.__class__.__name__

    @abstractmethod
    def add_processor(self, processor: BaseDataProcessor) -> BasePipeline:
        """
        Register a data processor stage into the pipeline.

        Parameters
        ----------
        processor : BaseDataProcessor
            The processor stage to add.

        Returns
        -------
        BasePipeline
            Self for method chaining.
        """
        pass

    @abstractmethod
    def execute(self, initial_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Execute the pipeline stages in sequential order.

        Parameters
        ----------
        initial_df : Optional[pd.DataFrame], default None
            Initial DataFrame to process. If None, loaded via attached loader.

        Returns
        -------
        pd.DataFrame
            Final processed DataFrame after executing all stages.
        """
        pass


class DataPipeline(BasePipeline):
    """
    Object-Oriented End-to-End Data Pipeline.

    Orchestrates data ingestion through BaseDataLoader and chains sequential
    BaseDataProcessor stages with detailed execution metrics and audit logging.
    """

    def __init__(
        self,
        loader: Optional[BaseDataLoader] = None,
        processors: Optional[Sequence[BaseDataProcessor]] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize the DataPipeline.

        Parameters
        ----------
        loader : Optional[BaseDataLoader], default None
            DataLoader responsible for reading raw datasets.
        processors : Optional[Sequence[BaseDataProcessor]], default None
            Initial sequence of processor stages.
        name : Optional[str], default None
            Custom identifier name.
        """
        super().__init__(name=name)
        self.loader: Optional[BaseDataLoader] = loader
        self._processors: List[BaseDataProcessor] = list(processors) if processors else []
        self._execution_summary: List[Dict[str, Any]] = []

    @property
    def processors(self) -> List[BaseDataProcessor]:
        """Return a defensive copy of registered processor stages."""
        return list(self._processors)

    @property
    def execution_summary(self) -> List[Dict[str, Any]]:
        """Return telemetry and execution records for the most recent run."""
        return list(self._execution_summary)

    def set_loader(self, loader: BaseDataLoader) -> DataPipeline:
        """
        Bind or replace the pipeline's data loader.

        Parameters
        ----------
        loader : BaseDataLoader
            The data loader instance.

        Returns
        -------
        DataPipeline
            self for method chaining.
        """
        if not isinstance(loader, BaseDataLoader):
            raise TypeError(f"Expected BaseDataLoader instance, got {type(loader).__name__}")
        self.loader = loader
        return self

    def add_processor(self, processor: BaseDataProcessor) -> DataPipeline:
        """
        Append a processor stage to the sequential processing chain.

        Parameters
        ----------
        processor : BaseDataProcessor
            The processor to append.

        Returns
        -------
        DataPipeline
            self for method chaining.
        """
        if not isinstance(processor, BaseDataProcessor):
            raise TypeError(f"Expected BaseDataProcessor instance, got {type(processor).__name__}")
        self._processors.append(processor)
        return self

    def add_validator(
        self,
        required_columns: Optional[Sequence[str]] = None,
        column_types: Optional[Mapping[str, Union[type, str]]] = None,
        bounds: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = None,
        non_null_columns: Optional[Sequence[str]] = None,
        unique_columns: Optional[Union[str, Sequence[str]]] = None,
        allowed_values: Optional[Mapping[str, Sequence[Any]]] = None,
        max_missing_ratio: float = 0.0,
        allow_empty: bool = False,
        name: Optional[str] = None,
    ) -> DataPipeline:
        """
        Register a configured DataValidator stage into the pipeline chain.

        Parameters
        ----------
        required_columns : Optional[Sequence[str]], default None
            Required columns.
        column_types : Optional[Mapping[str, Union[type, str]]], default None
            Expected column types.
        bounds : Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]], default None
            Numeric range boundaries.
        non_null_columns : Optional[Sequence[str]], default None
            Columns that must not contain missing values.
        unique_columns : Optional[Union[str, Sequence[str]]], default None
            Columns that must contain unique values.
        allowed_values : Optional[Mapping[str, Sequence[Any]]], default None
            Categorical allowed values.
        max_missing_ratio : float, default 0.0
            Maximum allowed missing ratio.
        allow_empty : bool, default False
            Whether empty DataFrames are allowed.
        name : Optional[str], default None
            Stage name.

        Returns
        -------
        DataPipeline
            self for method chaining.
        """
        validator = DataValidator(
            required_columns=required_columns,
            column_types=column_types,
            bounds=bounds,
            non_null_columns=non_null_columns,
            unique_columns=unique_columns,
            allowed_values=allowed_values,
            max_missing_ratio=max_missing_ratio,
            allow_empty=allow_empty,
            name=name or "PipelineDataValidator",
        )
        return self.add_processor(validator)

    def clear_processors(self) -> DataPipeline:
        """Clear all registered processor stages."""
        self._processors.clear()
        return self

    def __len__(self) -> int:
        """Return the number of registered processor stages."""
        return len(self._processors)

    def __iter__(self) -> Iterator[BaseDataProcessor]:
        """Iterate through registered processor stages."""
        return iter(self._processors)

    def __getitem__(self, index: int) -> BaseDataProcessor:
        """Retrieve a processor stage by index."""
        return self._processors[index]

    def execute(self, initial_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Execute the end-to-end data pipeline.

        Ingests data from loader (if initial_df is not provided), then pipes
        the dataset through each registered processor stage in sequence,
        recording runtime metrics.

        Parameters
        ----------
        initial_df : Optional[pd.DataFrame], default None
            Optional starting DataFrame. If None, `self.loader.load()` is called.

        Returns
        -------
        pd.DataFrame
            The processed DataFrame.

        Raises
        ------
        ValueError
            If no initial DataFrame is provided and no loader is attached.
        """
        self._execution_summary.clear()

        # Step 0: Ingestion
        current_df: pd.DataFrame
        if initial_df is not None:
            if not isinstance(initial_df, pd.DataFrame):
                raise TypeError(f"Expected pd.DataFrame, got {type(initial_df).__name__}")
            current_df = initial_df.copy()
            self._execution_summary.append({
                "stage": "Ingestion",
                "component": "Explicit DataFrame",
                "input_shape": list(current_df.shape),
                "output_shape": list(current_df.shape),
                "duration_seconds": 0.0,
            })
        elif self.loader is not None:
            start_time = time.perf_counter()
            current_df = self.loader.load()
            duration = time.perf_counter() - start_time
            self._execution_summary.append({
                "stage": "Ingestion",
                "component": self.loader.__class__.__name__,
                "input_shape": None,
                "output_shape": list(current_df.shape),
                "duration_seconds": round(duration, 6),
            })
        else:
            raise ValueError("No input DataFrame provided and pipeline has no configured loader.")

        # Step 1..N: Processor Execution
        for idx, processor in enumerate(self._processors, start=1):
            input_shape = list(current_df.shape)
            start_time = time.perf_counter()
            current_df = processor.process(current_df)
            duration = time.perf_counter() - start_time
            output_shape = list(current_df.shape)

            self._execution_summary.append({
                "stage": f"Step_{idx}_{processor.name}",
                "component": processor.__class__.__name__,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "duration_seconds": round(duration, 6),
            })

        return current_df

    def validate(
        self,
        df: Optional[pd.DataFrame] = None,
        required_columns: Optional[Sequence[str]] = None,
        column_types: Optional[Mapping[str, Union[type, str]]] = None,
        non_null_columns: Optional[Sequence[str]] = None,
        unique_columns: Optional[Union[str, Sequence[str]]] = None,
        bounds: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = None,
        allowed_values: Optional[Mapping[str, Sequence[Any]]] = None,
        max_missing_ratio: float = 0.0,
        allow_empty: bool = False,
        raise_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate an explicit DataFrame or data loaded from pipeline's loader.

        Parameters
        ----------
        df : Optional[pd.DataFrame], default None
            Target DataFrame. If None, loaded via self.loader.load().
        required_columns : Optional[Sequence[str]], default None
            Required column names.
        column_types : Optional[Mapping[str, Union[type, str]]], default None
            Expected column data types.
        non_null_columns : Optional[Sequence[str]], default None
            Columns that must not contain missing values.
        unique_columns : Optional[Union[str, Sequence[str]]], default None
            Columns that must contain unique values.
        bounds : Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]], default None
            Numeric range boundaries.
        allowed_values : Optional[Mapping[str, Sequence[Any]]], default None
            Permitted categorical values.
        max_missing_ratio : float, default 0.0
            Maximum allowable missing ratio.
        allow_empty : bool, default False
            Whether empty DataFrames are allowed.
        raise_on_error : bool, default True
            Whether to raise on violation or return diagnostic dict.

        Returns
        -------
        Dict[str, Any]
            Detailed validation diagnostic report.
        """
        target_df: pd.DataFrame
        if df is not None:
            target_df = df
        elif self.loader is not None:
            target_df = self.loader.load()
        else:
            raise ValueError("No input DataFrame provided and pipeline has no configured loader.")

        start_time = time.perf_counter()
        report = validate_pipeline_data(
            df=target_df,
            required_columns=required_columns,
            column_types=column_types,
            non_null_columns=non_null_columns,
            unique_columns=unique_columns,
            bounds=bounds,
            allowed_values=allowed_values,
            max_missing_ratio=max_missing_ratio,
            allow_empty=allow_empty,
            raise_on_error=raise_on_error,
        )
        duration = time.perf_counter() - start_time

        self._execution_summary.append({
            "stage": "Validation",
            "component": "validate_pipeline_data",
            "input_shape": list(target_df.shape),
            "output_shape": list(target_df.shape),
            "duration_seconds": round(duration, 6),
            "is_valid": report["is_valid"],
            "checks_performed": report["checks_performed"],
        })
        return report


# ============================================================================
# Section 2: Pipeline Validation Functions & Factory Helpers
# ============================================================================

def validate_pipeline_data(
    df: pd.DataFrame,
    required_columns: Optional[Sequence[str]] = None,
    column_types: Optional[Mapping[str, Union[type, str]]] = None,
    non_null_columns: Optional[Sequence[str]] = None,
    unique_columns: Optional[Union[str, Sequence[str]]] = None,
    bounds: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = None,
    allowed_values: Optional[Mapping[str, Sequence[Any]]] = None,
    max_missing_ratio: float = 0.0,
    allow_empty: bool = False,
    raise_on_error: bool = True,
) -> Dict[str, Any]:
    """
    Validate tabular data entering, traversing, or exiting the data pipeline.

    Applies comprehensive validation across structural schema integrity,
    null value constraints, uniqueness/key constraints, numeric boundaries,
    and categorical domain restrictions with detailed validation diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to validate.
    required_columns : Optional[Sequence[str]], default None
        Sequence of column names that must exist in df.
    column_types : Optional[Mapping[str, Union[type, str]]], default None
        Mapping of column names to expected types or dtype strings.
    non_null_columns : Optional[Sequence[str]], default None
        Columns that must satisfy missing value constraints (ratio <= max_missing_ratio).
    unique_columns : Optional[Union[str, Sequence[str]]], default None
        Column name or sequence of columns forming a primary/composite key
        that must have strictly unique (non-duplicated) values.
    bounds : Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]], default None
        Mapping from column name to (min_val, max_val) numeric range tuple.
    allowed_values : Optional[Mapping[str, Sequence[Any]]], default None
        Mapping from column name to allowed set/list of values.
    max_missing_ratio : float, default 0.0
        Maximum allowable missing/null ratio for non_null_columns (0.0 = zero nulls).
    allow_empty : bool, default False
        Whether an empty DataFrame (0 rows) is considered valid.
    raise_on_error : bool, default True
        If True, raises TypeError or ValueError immediately on first constraint failure.
        If False, collects all errors and returns a structured validation diagnostics report.

    Returns
    -------
    Dict[str, Any]
        Structured validation diagnostics report containing:
        - 'is_valid' (bool): True if all checks passed.
        - 'total_rows' (int): Total rows evaluated.
        - 'total_columns' (int): Total columns evaluated.
        - 'checks_performed' (List[str]): List of validation checks executed.
        - 'errors' (List[str]): Explanations of any validation failures.
        - 'details' (Dict[str, Any]): Breakdown per check.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame, or if type constraints mismatch (when raise_on_error=True).
    ValueError
        If schema, range, null, or uniqueness constraints fail (when raise_on_error=True).
    """
    if not isinstance(df, pd.DataFrame):
        if raise_on_error:
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
        return {
            "is_valid": False,
            "total_rows": 0,
            "total_columns": 0,
            "checks_performed": ["type_check"],
            "errors": [f"Expected pd.DataFrame, got {type(df).__name__}"],
            "details": {},
        }

    checks_performed: List[str] = []
    errors: List[str] = []
    details: Dict[str, Any] = {}

    # Check 1: Empty DataFrame check
    checks_performed.append("empty_check")
    if not allow_empty and df.empty:
        err_msg = "DataFrame is empty, which violates the schema constraint (allow_empty=False)."
        errors.append(err_msg)
        if raise_on_error:
            raise ValueError(err_msg)
    details["allow_empty"] = allow_empty
    details["is_empty"] = df.empty

    # Check 2: Schema (required columns and column types)
    if required_columns is not None or column_types is not None:
        checks_performed.append("schema_validation")
        try:
            validate_schema(
                df,
                required_columns=required_columns or [],
                column_types=column_types,
                allow_empty=allow_empty,
            )
            details["schema"] = {"status": "passed"}
        except (TypeError, ValueError) as err:
            details["schema"] = {"status": "failed", "error": str(err)}
            errors.append(str(err))
            if raise_on_error:
                raise

    # Check 3: Non-null constraints
    if non_null_columns is not None:
        checks_performed.append("null_constraints")
        try:
            null_res = validate_null_constraints(
                df,
                columns=non_null_columns,
                max_missing_ratio=max_missing_ratio,
                raise_on_error=raise_on_error,
            )
            details["null_constraints"] = null_res
            failed_null = [col for col, ok in null_res.items() if not ok]
            if failed_null:
                err_msg = f"Null constraint violated for columns: {failed_null}"
                errors.append(err_msg)
                if raise_on_error:
                    raise ValueError(err_msg)
        except (TypeError, ValueError, KeyError) as err:
            details["null_constraints"] = {"error": str(err)}
            errors.append(str(err))
            if raise_on_error:
                raise

    # Check 4: Uniqueness constraints
    if unique_columns is not None:
        checks_performed.append("unique_constraints")
        try:
            is_unique = validate_unique_constraints(
                df,
                columns=unique_columns,
                allow_na=False,
                raise_on_error=raise_on_error,
            )
            details["unique_constraints"] = {
                "columns": unique_columns if isinstance(unique_columns, str) else list(unique_columns),
                "is_unique": is_unique,
            }
            if not is_unique:
                err_msg = f"Uniqueness constraint violated for column(s): {unique_columns}"
                errors.append(err_msg)
                if raise_on_error:
                    raise ValueError(err_msg)
        except (TypeError, ValueError, KeyError) as err:
            details["unique_constraints"] = {"error": str(err)}
            errors.append(str(err))
            if raise_on_error:
                raise

    # Check 5: Numerical range bounds
    if bounds is not None:
        checks_performed.append("range_bounds")
        try:
            range_res = validate_column_ranges(df, bounds=bounds)
            details["range_bounds"] = range_res
            failed_ranges = [col for col, ok in range_res.items() if not ok]
            if failed_ranges:
                err_msg = f"Numeric range validation failed for columns: {failed_ranges}"
                errors.append(err_msg)
                if raise_on_error:
                    raise ValueError(err_msg)
        except (TypeError, ValueError, KeyError) as err:
            details["range_bounds"] = {"error": str(err)}
            errors.append(str(err))
            if raise_on_error:
                raise

    # Check 6: Allowed categorical values
    if allowed_values is not None:
        checks_performed.append("allowed_values")
        try:
            allowed_res = validate_allowed_values(
                df,
                allowed_values=allowed_values,
                allow_null=True,
                raise_on_error=raise_on_error,
            )
            details["allowed_values"] = allowed_res
            failed_allowed = [col for col, ok in allowed_res.items() if not ok]
            if failed_allowed:
                err_msg = f"Allowed values constraint violated for columns: {failed_allowed}"
                errors.append(err_msg)
                if raise_on_error:
                    raise ValueError(err_msg)
        except (TypeError, ValueError, KeyError) as err:
            details["allowed_values"] = {"error": str(err)}
            errors.append(str(err))
            if raise_on_error:
                raise

    return {
        "is_valid": len(errors) == 0,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "checks_performed": checks_performed,
        "errors": errors,
        "details": details,
    }

def build_pipeline(
    loader: Optional[BaseDataLoader] = None,
    processors: Optional[Sequence[BaseDataProcessor]] = None,
    name: Optional[str] = None,
) -> DataPipeline:
    """
    Factory function to build a DataPipeline instance.

    Parameters
    ----------
    loader : Optional[BaseDataLoader], default None
        Optional loader.
    processors : Optional[Sequence[BaseDataProcessor]], default None
        Sequence of processors.
    name : Optional[str], default None
        Pipeline name.

    Returns
    -------
    DataPipeline
        Configured DataPipeline instance.
    """
    pipeline = DataPipeline(loader=loader, name=name)
    if processors:
        for proc in processors:
            pipeline.add_processor(proc)
    return pipeline


def create_standard_pipeline(
    file_path: Union[str, Path],
    required_columns: Sequence[str],
    filter_column: Optional[str] = None,
    filter_operator: Optional[str] = None,
    filter_value: Optional[Any] = None,
    normalize_columns: Optional[Sequence[str]] = None,
    normalize_method: str = "minmax",
    group_by: Optional[Union[str, Sequence[str]]] = None,
    aggregations: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
    non_null_columns: Optional[Sequence[str]] = None,
    unique_columns: Optional[Union[str, Sequence[str]]] = None,
) -> DataPipeline:
    """
    Factory function to assemble a standard production pipeline combining:
    1. DataLoader for CSV ingestion
    2. DataValidator for schema validation, null checks, and uniqueness constraints
    3. DataTransformer for optional filtering and normalization
    4. DataAggregator for optional grouping

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the CSV file.
    required_columns : Sequence[str]
        Required column names to validate.
    filter_column : Optional[str], default None
        Optional column to filter on.
    filter_operator : Optional[str], default None
        Filter operator (e.g., '>', '==', 'in').
    filter_value : Optional[Any], default None
        Comparison target value.
    normalize_columns : Optional[Sequence[str]], default None
        Numeric columns to scale.
    normalize_method : str, default "minmax"
        Normalization method ('minmax' or 'zscore').
    group_by : Optional[Union[str, Sequence[str]]], default None
        Grouping column(s).
    aggregations : Optional[Mapping[str, Union[str, Sequence[str]]]], default None
        Aggregation specifications.
    non_null_columns : Optional[Sequence[str]], default None
        Optional columns that must not contain missing values.
    unique_columns : Optional[Union[str, Sequence[str]]], default None
        Optional primary/unique key column(s).

    Returns
    -------
    DataPipeline
        Ready-to-execute DataPipeline instance.
    """
    loader = DataLoader(file_path=file_path)
    validator = DataValidator(
        required_columns=required_columns,
        non_null_columns=non_null_columns,
        unique_columns=unique_columns,
    )

    transformer = DataTransformer(
        normalize_cols=normalize_columns,
        norm_method=normalize_method,
    )
    if filter_column is not None and filter_operator is not None:
        transformer.add_filter(filter_column, filter_operator, filter_value)

    pipeline = DataPipeline(loader=loader, name="StandardProductionPipeline")
    pipeline.add_processor(validator)
    pipeline.add_processor(transformer)

    if group_by is not None and aggregations is not None:
        aggregator = DataAggregator(group_by=group_by, aggregations=aggregations)
        pipeline.add_processor(aggregator)

    return pipeline


if __name__ == "__main__":
    print("pipeline module loaded successfully.")

