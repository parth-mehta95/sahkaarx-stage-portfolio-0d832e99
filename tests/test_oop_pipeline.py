"""
tests/test_oop_pipeline.py

Comprehensive automated unit test suite for the Object-Oriented pipeline classes:
- BaseDataLoader, DataLoader, CSVDataLoader, JSONDataLoader
- BaseDataProcessor, DataValidator, DataTransformer, DataAggregator, DataProcessor
- BasePipeline, DataPipeline, build_pipeline, create_standard_pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from data_loader import BaseDataLoader, CSVDataLoader, DataLoader, JSONDataLoader
from data_processor import (
    BaseDataProcessor,
    DataAggregator,
    DataProcessor,
    DataTransformer,
    DataValidator,
)
from pipeline import BasePipeline, DataPipeline, build_pipeline, create_standard_pipeline


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Fixture providing a standard multi-column test DataFrame."""
    return pd.DataFrame({
        "employee_id": [101, 102, 103, 104, 105],
        "name": [" Alice ", "Bob", "Charlie ", "Diana", "Evan"],
        "department": ["Engineering", "Marketing", "Engineering", "HR", "Marketing"],
        "salary": [120000.0, 95000.0, 110000.0, 85000.0, 98000.0],
        "experience": [5, 3, 4, 2, 3],
    })


@pytest.fixture
def sample_csv_file(tmp_path: Path, sample_dataframe: pd.DataFrame) -> Path:
    """Fixture creating a temporary CSV file with test data."""
    csv_path = tmp_path / "employees.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_json_file(tmp_path: Path, sample_dataframe: pd.DataFrame) -> Path:
    """Fixture creating a temporary JSON file with test data."""
    json_path = tmp_path / "employees.json"
    sample_dataframe.to_json(json_path, orient="records", indent=2)
    return json_path


# ============================================================================
# Section 1: DataLoader Hierarchy Tests
# ============================================================================

class TestDataLoaderHierarchy:
    """Tests covering BaseDataLoader, DataLoader, CSVDataLoader, and JSONDataLoader."""

    def test_base_data_loader_is_abstract(self) -> None:
        """Verify that BaseDataLoader cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseDataLoader()  # type: ignore[abstract]

    def test_data_loader_initialization(self, sample_csv_file: Path) -> None:
        """Verify DataLoader initializes with default parameters and path."""
        loader = DataLoader(file_path=sample_csv_file)
        assert loader.file_path == sample_csv_file.resolve()
        assert loader.delimiter == ","
        assert loader.encoding == "utf-8"
        assert not loader.is_loaded
        assert loader.data is None

    def test_data_loader_load_and_cache(self, sample_csv_file: Path) -> None:
        """Verify loading CSV sets internal cached data and returns DataFrame."""
        loader = DataLoader(file_path=sample_csv_file)
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert loader.is_loaded
        assert loader.data is not None
        assert len(loader.data) == 5

    def test_data_loader_load_with_path_override(self, tmp_path: Path) -> None:
        """Verify load() accepts explicit path argument overriding bound path."""
        file_a = tmp_path / "a.csv"
        file_b = tmp_path / "b.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(file_a, index=False)
        pd.DataFrame({"x": [10, 20, 30]}).to_csv(file_b, index=False)

        loader = DataLoader(file_path=file_a)
        df = loader.load(source=file_b)
        assert len(df) == 3

    def test_data_loader_clean_cached_data(self, tmp_path: Path) -> None:
        """Verify clean() transforms cached data and updates internal state."""
        raw_csv = tmp_path / "dirty.csv"
        raw_csv.write_text("name, val \n Alice , 10\n,20\nBob,30\n", encoding="utf-8")

        loader = DataLoader(file_path=raw_csv)
        loader.load()
        cleaned = loader.clean(drop_na=True, strip_whitespace=True)
        assert len(cleaned) == 2
        assert "val" in cleaned.columns
        assert loader.data is not None
        assert len(loader.data) == 2

    def test_data_loader_clean_explicit_dataframe(self) -> None:
        """Verify clean() accepts explicit DataFrame without needing cached data."""
        loader = DataLoader()
        raw_df = pd.DataFrame({"col ": [" val1 ", "val2", None]})
        cleaned = loader.clean(df=raw_df, drop_na=True)
        assert len(cleaned) == 2
        assert list(cleaned.columns) == ["col"]

    def test_data_loader_clean_uninitialized_raises(self) -> None:
        """Verify clean() raises ValueError when no data is provided or loaded."""
        loader = DataLoader()
        with pytest.raises(ValueError, match="No DataFrame provided and loader contains no loaded data"):
            loader.clean()

    def test_data_loader_get_summary(self, sample_csv_file: Path) -> None:
        """Verify get_summary() produces structural metadata dictionary."""
        loader = DataLoader(file_path=sample_csv_file)
        loader.load()
        summary = loader.get_summary()
        assert summary["num_rows"] == 5
        assert summary["num_columns"] == 5
        assert "department" in summary["columns"]

    def test_data_loader_save_cached_and_explicit(self, tmp_path: Path, sample_dataframe: pd.DataFrame) -> None:
        """Verify save() writes cached data or explicit DataFrame to disk."""
        dest_csv = tmp_path / "out" / "saved.csv"
        loader = DataLoader()
        out_path = loader.save(df=sample_dataframe, destination=dest_csv)
        assert out_path.exists()

        # Reload to verify content
        reloaded = pd.read_csv(out_path)
        assert len(reloaded) == len(sample_dataframe)

    def test_data_loader_error_handling(self, tmp_path: Path) -> None:
        """Verify DataLoader error handling for missing files and directories."""
        loader = DataLoader(file_path=tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            loader.load()

        dir_path = tmp_path / "some_dir"
        dir_path.mkdir()
        loader_dir = DataLoader(file_path=dir_path)
        with pytest.raises(ValueError, match="Expected a file path, but got a directory"):
            loader_dir.load()

    def test_csv_data_loader_alias(self) -> None:
        """Verify CSVDataLoader is an exact alias for DataLoader."""
        assert CSVDataLoader is DataLoader

    def test_json_data_loader_inheritance_and_polymorphism(
        self, sample_json_file: Path, tmp_path: Path
    ) -> None:
        """Verify JSONDataLoader inherits BaseDataLoader and supports polymorphism."""
        loader = JSONDataLoader(file_path=sample_json_file, orient="records")
        assert isinstance(loader, BaseDataLoader)

        df = loader.load()
        assert len(df) == 5
        assert loader.is_loaded

        # Test JSON save
        dest_json = tmp_path / "exported.json"
        saved = loader.save(destination=dest_json, orient="records")
        assert saved.exists()


# ============================================================================
# Section 2: DataProcessor Hierarchy Tests
# ============================================================================

class TestDataProcessorHierarchy:
    """Tests covering BaseDataProcessor, DataValidator, DataTransformer, DataAggregator."""

    def test_base_data_processor_is_abstract(self) -> None:
        """Verify that BaseDataProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseDataProcessor()  # type: ignore[abstract]

    def test_data_processor_callable_interface(self, sample_dataframe: pd.DataFrame) -> None:
        """Verify processor can be invoked directly as a callable."""
        processor = DataProcessor()
        result = processor(sample_dataframe)
        assert len(result) == len(sample_dataframe)

    def test_data_validator_schema_and_range(self, sample_dataframe: pd.DataFrame) -> None:
        """Verify DataValidator validates schema and numerical ranges."""
        validator = DataValidator(
            required_columns=["employee_id", "salary"],
            column_types={"salary": float},
            bounds={"salary": (50000.0, 200000.0)},
        )
        assert isinstance(validator, BaseDataProcessor)

        # Process passes valid DataFrame
        validated_df = validator.process(sample_dataframe)
        assert len(validated_df) == len(sample_dataframe)
        assert len(validator.audit_log) > 0

        # Process raises when schema is missing column
        bad_df = sample_dataframe.drop(columns=["salary"])
        with pytest.raises(ValueError, match="missing required columns"):
            validator.process(bad_df)

        # Process raises when numeric bounds are violated
        out_of_bounds_df = sample_dataframe.copy()
        out_of_bounds_df.loc[0, "salary"] = 10000.0
        with pytest.raises(ValueError, match="Numeric range validation failed"):
            validator.process(out_of_bounds_df)

    def test_data_transformer_pipeline(self, sample_dataframe: pd.DataFrame) -> None:
        """Verify DataTransformer applies filters, normalization, and casting."""
        transformer = DataTransformer(
            filters=[("department", "==", "Engineering")],
            normalize_cols=["salary"],
            norm_method="minmax",
            cast_types={"employee_id": "int64"},
        )
        assert isinstance(transformer, BaseDataProcessor)

        result = transformer.process(sample_dataframe)
        assert len(result) == 2  # Only Engineering rows
        assert result["salary"].min() >= 0.0
        assert result["salary"].max() <= 1.0

    def test_data_aggregator_group_and_stats(self, sample_dataframe: pd.DataFrame) -> None:
        """Verify DataAggregator groups metrics and computes summary statistics."""
        aggregator = DataAggregator(
            group_by="department",
            aggregations={"salary": ["mean", "sum"]},
        )
        assert isinstance(aggregator, BaseDataProcessor)

        agg_df = aggregator.process(sample_dataframe)
        assert len(agg_df) == 3  # 3 distinct departments
        assert "salary_mean" in agg_df.columns
        assert "salary_sum" in agg_df.columns

        # Test summary stats
        stats = aggregator.calculate_summary_statistics(sample_dataframe, columns=["salary"])
        assert "salary" in stats
        assert stats["salary"]["count"] == 5.0


# ============================================================================
# Section 3: Unified DataProcessor Tests
# ============================================================================

class TestUnifiedDataProcessor:
    """Tests covering unified DataProcessor fluent API and method organization."""

    def test_dataprocessor_initialization_and_binding(self, sample_dataframe: pd.DataFrame) -> None:
        """Verify DataProcessor initializes and binds DataFrames."""
        processor = DataProcessor(df=sample_dataframe)
        assert processor.has_data
        assert processor.data is not None
        assert len(processor.data) == 5

    def test_dataprocessor_fluent_method_chaining(self, sample_dataframe: pd.DataFrame) -> None:
        """Verify fluent method chaining across filter, normalize, and cast."""
        processor = DataProcessor()
        result_df = (
            processor.set_data(sample_dataframe)
            .filter("salary", ">=", 95000.0)
            .normalize(["salary"], method="minmax")
            .cast({"experience": "int64"})
            .get_data()
        )
        assert len(result_df) == 4
        assert result_df["salary"].min() >= 0.0
        assert result_df["salary"].max() <= 1.0
        assert len(processor.audit_log) >= 4

    def test_dataprocessor_run_pipeline(self, sample_dataframe: pd.DataFrame) -> None:
        """Verify DataProcessor.run_pipeline executes full composite pipeline."""
        processor = DataProcessor(df=sample_dataframe)
        result = processor.run_pipeline(
            required_columns=["department", "salary"],
            filter_column="experience",
            filter_op=">=",
            filter_value=3,
            normalize_columns=["salary"],
            group_by="department",
            aggregations={"salary": ["mean"]},
        )
        assert isinstance(result, pd.DataFrame)
        assert "department" in result.columns
        assert "salary_mean" in result.columns

    def test_dataprocessor_audit_logging(self, sample_dataframe: pd.DataFrame) -> None:
        """Verify audit log records each action with details and can be cleared."""
        processor = DataProcessor(df=sample_dataframe)
        processor.filter("experience", ">", 2)
        assert len(processor.audit_log) >= 1
        processor.reset_audit_log()
        assert len(processor.audit_log) == 0


# ============================================================================
# Section 4: DataPipeline Orchestration Tests
# ============================================================================

class TestDataPipelineOrchestration:
    """Tests covering BasePipeline, DataPipeline, and factory functions."""

    def test_base_pipeline_is_abstract(self) -> None:
        """Verify BasePipeline cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BasePipeline()  # type: ignore[abstract]

    def test_data_pipeline_with_explicit_dataframe(self, sample_dataframe: pd.DataFrame) -> None:
        """Verify DataPipeline execution with explicit DataFrame and multiple stages."""
        pipeline = DataPipeline(name="TestPipeline")
        validator = DataValidator(required_columns=["department", "salary"])
        transformer = DataTransformer()
        transformer.add_filter("experience", ">=", 3)
        aggregator = DataAggregator(
            group_by="department",
            aggregations={"salary": ["mean"]},
        )

        pipeline.add_processor(validator)
        pipeline.add_processor(transformer)
        pipeline.add_processor(aggregator)

        assert len(pipeline) == 3
        assert pipeline[0] is validator

        result = pipeline.execute(initial_df=sample_dataframe)
        assert len(result) == 2  # Engineering and Marketing
        assert "salary_mean" in result.columns

        # Check telemetry
        summary = pipeline.execution_summary
        assert len(summary) == 4  # Ingestion + 3 steps
        assert summary[0]["stage"] == "Ingestion"
        assert summary[1]["stage"] == f"Step_1_{validator.name}"

    def test_data_pipeline_with_loader(self, sample_csv_file: Path) -> None:
        """Verify DataPipeline ingests data through bound BaseDataLoader."""
        loader = DataLoader(file_path=sample_csv_file)
        validator = DataValidator(required_columns=["employee_id", "salary"])
        pipeline = build_pipeline(loader=loader, processors=[validator])

        result = pipeline.execute()
        assert len(result) == 5
        summary = pipeline.execution_summary
        assert summary[0]["component"] == "DataLoader"

    def test_create_standard_pipeline_factory(self, sample_csv_file: Path) -> None:
        """Verify create_standard_pipeline factory builds and executes properly."""
        pipeline = create_standard_pipeline(
            file_path=sample_csv_file,
            required_columns=["department", "salary"],
            filter_column="salary",
            filter_operator=">=",
            filter_value=95000.0,
            normalize_columns=["salary"],
            group_by="department",
            aggregations={"salary": ["mean"]},
        )
        assert len(pipeline) == 3
        result = pipeline.execute()
        assert "salary_mean" in result.columns
        assert len(result) > 0

    def test_data_pipeline_type_error_on_invalid_processor(self) -> None:
        """Verify TypeError when adding an object not inheriting BaseDataProcessor."""
        pipeline = DataPipeline()
        with pytest.raises(TypeError, match="Expected BaseDataProcessor instance"):
            pipeline.add_processor("not_a_processor")  # type: ignore[arg-type]

    def test_data_pipeline_missing_source_raises(self) -> None:
        """Verify ValueError when executing pipeline without loader or initial_df."""
        pipeline = DataPipeline()
        with pytest.raises(ValueError, match="No input DataFrame provided and pipeline has no configured loader"):
            pipeline.execute()
