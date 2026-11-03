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
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union
import pandas as pd  # type: ignore[import-untyped]

from data_loader import BaseDataLoader, DataLoader
from data_processor import (
    BaseDataProcessor,
    DataAggregator,
    DataProcessor,
    DataTransformer,
    DataValidator,
)

__all__ = [
    "BasePipeline",
    "DataPipeline",
    "build_pipeline",
    "create_standard_pipeline",
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


# ============================================================================
# Section 2: Pipeline Factory Functions & Helpers
# ============================================================================

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
) -> DataPipeline:
    """
    Factory function to assemble a standard production pipeline combining:
    1. DataLoader for CSV ingestion
    2. DataValidator for schema validation
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

    Returns
    -------
    DataPipeline
        Ready-to-execute DataPipeline instance.
    """
    loader = DataLoader(file_path=file_path)
    validator = DataValidator(required_columns=required_columns)

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

