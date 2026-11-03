"""
tests/test_data_validation.py

Comprehensive test suite verifying type-hinted data validation functions,
including null constraint verification, uniqueness verification, categorical domain checks,
and pipeline-level orchestration validation.
"""

from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
import pytest

from data_processor import (
    DataProcessor,
    DataValidator,
    validate_allowed_values,
    validate_null_constraints,
    validate_unique_constraints,
)
from pipeline import (
    DataPipeline,
    create_standard_pipeline,
    validate_pipeline_data,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Fixture providing clean tabular data."""
    return pd.DataFrame({
        "user_id": [101, 102, 103, 104, 105],
        "username": ["alice", "bob", "carol", "dave", "eve"],
        "age": [25, 30, 35, 40, 45],
        "score": [88.5, 92.0, 79.5, 95.0, 84.0],
        "status": ["active", "pending", "active", "active", "inactive"],
    })


@pytest.fixture
def sample_dirty_df() -> pd.DataFrame:
    """Fixture providing tabular data containing nulls, duplicates, and invalid categories."""
    return pd.DataFrame({
        "user_id": [101, 102, 102, 104, 105],  # duplicate 102
        "username": ["alice", None, "carol", "dave", "eve"],  # null username
        "age": [25, 30, -5, 40, 150],  # out of bound ages
        "score": [88.5, None, 79.5, None, 84.0],  # 40% missing scores
        "status": ["active", "unknown", "active", "deleted", "inactive"],  # invalid statuses
    })


# ============================================================================
# Section 1: validate_null_constraints Tests
# ============================================================================

class TestValidateNullConstraints:
    """Unit tests for validate_null_constraints."""

    def test_clean_data_all_columns_pass(self, sample_valid_df: pd.DataFrame) -> None:
        result = validate_null_constraints(sample_valid_df)
        assert all(result.values())
        assert set(result.keys()) == set(sample_valid_df.columns)

    def test_clean_data_specific_columns(self, sample_valid_df: pd.DataFrame) -> None:
        result = validate_null_constraints(sample_valid_df, columns=["user_id", "username"])
        assert result == {"user_id": True, "username": True}

    def test_nulls_detected_without_raising(self, sample_dirty_df: pd.DataFrame) -> None:
        result = validate_null_constraints(sample_dirty_df, columns=["user_id", "username", "score"])
        assert result["user_id"] is True
        assert result["username"] is False
        assert result["score"] is False

    def test_nulls_raise_on_error(self, sample_dirty_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Null constraint validation failed for columns"):
            validate_null_constraints(sample_dirty_df, columns=["username"], raise_on_error=True)

    def test_max_missing_ratio_threshold(self, sample_dirty_df: pd.DataFrame) -> None:
        # score has 2 nulls out of 5 = 40% (0.40)
        assert validate_null_constraints(sample_dirty_df, columns=["score"], max_missing_ratio=0.5)["score"] is True
        assert validate_null_constraints(sample_dirty_df, columns=["score"], max_missing_ratio=0.3)["score"] is False

    def test_empty_dataframe(self) -> None:
        empty_df = pd.DataFrame(columns=["a", "b"])
        result = validate_null_constraints(empty_df)
        assert result == {"a": True, "b": True}

    def test_invalid_dataframe_type(self) -> None:
        with pytest.raises(TypeError, match="Expected input of type pd.DataFrame"):
            validate_null_constraints([1, 2, 3])  # type: ignore[arg-type]

    def test_invalid_max_missing_ratio_type(self, sample_valid_df: pd.DataFrame) -> None:
        with pytest.raises(TypeError, match="Expected numeric max_missing_ratio"):
            validate_null_constraints(sample_valid_df, max_missing_ratio="zero")  # type: ignore[arg-type]

    def test_invalid_max_missing_ratio_value(self, sample_valid_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="max_missing_ratio must be between 0.0 and 1.0"):
            validate_null_constraints(sample_valid_df, max_missing_ratio=1.5)

    def test_missing_column_raises_key_error(self, sample_valid_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="not found in DataFrame"):
            validate_null_constraints(sample_valid_df, columns=["non_existent_column"])


# ============================================================================
# Section 2: validate_unique_constraints Tests
# ============================================================================

class TestValidateUniqueConstraints:
    """Unit tests for validate_unique_constraints."""

    def test_unique_single_column_passes(self, sample_valid_df: pd.DataFrame) -> None:
        assert validate_unique_constraints(sample_valid_df, columns="user_id") is True
        assert validate_unique_constraints(sample_valid_df, columns=["user_id"]) is True

    def test_duplicate_single_column_returns_false(self, sample_dirty_df: pd.DataFrame) -> None:
        assert validate_unique_constraints(sample_dirty_df, columns="user_id") is False

    def test_duplicate_raises_on_error(self, sample_dirty_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Uniqueness constraint validation failed"):
            validate_unique_constraints(sample_dirty_df, columns="user_id", raise_on_error=True)

    def test_composite_unique_key(self) -> None:
        df = pd.DataFrame({
            "first_name": ["John", "John", "Jane"],
            "last_name": ["Doe", "Smith", "Doe"],
            "age": [30, 25, 28],
        })
        # first_name has duplicates, but (first_name, last_name) is unique
        assert validate_unique_constraints(df, columns="first_name") is False
        assert validate_unique_constraints(df, columns=["first_name", "last_name"]) is True

    def test_composite_duplicate_key(self) -> None:
        df = pd.DataFrame({
            "first_name": ["John", "John"],
            "last_name": ["Doe", "Doe"],
        })
        assert validate_unique_constraints(df, columns=["first_name", "last_name"]) is False

    def test_null_values_with_allow_na(self) -> None:
        df = pd.DataFrame({
            "id": [1, 2, None],
        })
        # with allow_na=False, nulls fail uniqueness
        assert validate_unique_constraints(df, columns="id", allow_na=False) is False
        # with allow_na=True, remaining non-nulls [1, 2] are unique
        assert validate_unique_constraints(df, columns="id", allow_na=True) is True

    def test_empty_dataframe(self) -> None:
        empty_df = pd.DataFrame(columns=["id", "name"])
        assert validate_unique_constraints(empty_df, columns="id") is True

    def test_empty_columns_raises_value_error(self, sample_valid_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="columns cannot be empty"):
            validate_unique_constraints(sample_valid_df, columns=[])

    def test_missing_column_raises_key_error(self, sample_valid_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="not found in DataFrame"):
            validate_unique_constraints(sample_valid_df, columns="ghost_col")

    def test_invalid_dataframe_type(self) -> None:
        with pytest.raises(TypeError, match="Expected input of type pd.DataFrame"):
            validate_unique_constraints({"id": [1, 2]}, columns="id")  # type: ignore[arg-type]


# ============================================================================
# Section 3: validate_allowed_values Tests
# ============================================================================

class TestValidateAllowedValues:
    """Unit tests for validate_allowed_values."""

    def test_valid_allowed_values_pass(self, sample_valid_df: pd.DataFrame) -> None:
        result = validate_allowed_values(
            sample_valid_df,
            allowed_values={"status": ["active", "pending", "inactive"]},
        )
        assert result == {"status": True}

    def test_invalid_allowed_values_return_false(self, sample_dirty_df: pd.DataFrame) -> None:
        result = validate_allowed_values(
            sample_dirty_df,
            allowed_values={"status": ["active", "pending", "inactive"]},
        )
        assert result == {"status": False}

    def test_invalid_allowed_values_raise_on_error(self, sample_dirty_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Allowed values constraint violated"):
            validate_allowed_values(
                sample_dirty_df,
                allowed_values={"status": ["active", "pending", "inactive"]},
                raise_on_error=True,
            )

    def test_missing_column_raises_key_error(self, sample_valid_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="not found in DataFrame"):
            validate_allowed_values(sample_valid_df, allowed_values={"missing": [1, 2]})

    def test_empty_mapping_raises_value_error(self, sample_valid_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="allowed_values mapping cannot be empty"):
            validate_allowed_values(sample_valid_df, allowed_values={})


# ============================================================================
# Section 4: validate_pipeline_data Tests
# ============================================================================

class TestValidatePipelineData:
    """Unit tests for top-level validate_pipeline_data in pipeline.py."""

    def test_comprehensive_validation_passes(self, sample_valid_df: pd.DataFrame) -> None:
        report = validate_pipeline_data(
            sample_valid_df,
            required_columns=["user_id", "username", "age", "score", "status"],
            column_types={"user_id": int, "score": float},
            non_null_columns=["user_id", "username"],
            unique_columns="user_id",
            bounds={"age": (18.0, 100.0), "score": (0.0, 100.0)},
            allowed_values={"status": ["active", "pending", "inactive"]},
            raise_on_error=True,
        )
        assert report["is_valid"] is True
        assert report["total_rows"] == 5
        assert len(report["errors"]) == 0
        assert "null_constraints" in report["checks_performed"]
        assert "unique_constraints" in report["checks_performed"]

    def test_comprehensive_validation_collects_errors(self, sample_dirty_df: pd.DataFrame) -> None:
        report = validate_pipeline_data(
            sample_dirty_df,
            required_columns=["user_id", "username", "age", "score", "status"],
            non_null_columns=["username"],
            unique_columns="user_id",
            bounds={"age": (0.0, 100.0)},
            allowed_values={"status": ["active", "pending", "inactive"]},
            raise_on_error=False,
        )
        assert report["is_valid"] is False
        assert len(report["errors"]) > 0
        assert any("Null constraint" in err for err in report["errors"])
        assert any("Uniqueness constraint" in err for err in report["errors"])
        assert any("Numeric range" in err for err in report["errors"])

    def test_raise_on_error_immediately_raises(self, sample_dirty_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            validate_pipeline_data(
                sample_dirty_df,
                non_null_columns=["username"],
                raise_on_error=True,
            )

    def test_empty_dataframe_not_allowed(self) -> None:
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_pipeline_data(empty_df, allow_empty=False, raise_on_error=True)

    def test_invalid_type_handled(self) -> None:
        report = validate_pipeline_data("not a df", raise_on_error=False)  # type: ignore[arg-type]
        assert report["is_valid"] is False
        assert "Expected pd.DataFrame" in report["errors"][0]


# ============================================================================
# Section 5: DataValidator & DataPipeline Integration Tests
# ============================================================================

class TestValidationOOPIntegration:
    """Tests verifying DataValidator, DataProcessor, and DataPipeline integration."""

    def test_datavalidator_with_extended_constraints(self, sample_valid_df: pd.DataFrame) -> None:
        validator = DataValidator(
            required_columns=["user_id", "username"],
            non_null_columns=["user_id", "username"],
            unique_columns="user_id",
            allowed_values={"status": ["active", "pending", "inactive"]},
            bounds={"age": (0.0, 120.0)},
        )
        processed = validator.process(sample_valid_df)
        assert len(processed) == 5
        assert len(validator.audit_log) >= 4

    def test_datavalidator_raises_on_violations(self, sample_dirty_df: pd.DataFrame) -> None:
        validator = DataValidator(non_null_columns=["username"])
        with pytest.raises(ValueError, match="Null constraint validation failed"):
            validator.process(sample_dirty_df)

    def test_datapipeline_add_validator_fluent(self, sample_valid_df: pd.DataFrame) -> None:
        pipeline = DataPipeline()
        pipeline.add_validator(
            required_columns=["user_id", "username"],
            non_null_columns=["username"],
            unique_columns="user_id",
        )
        assert len(pipeline) == 1
        result = pipeline.execute(sample_valid_df)
        assert len(result) == 5
        summary = pipeline.execution_summary
        assert len(summary) == 2  # Ingestion + Step_1_PipelineDataValidator

    def test_datapipeline_validate_method(self, sample_valid_df: pd.DataFrame) -> None:
        pipeline = DataPipeline()
        report = pipeline.validate(
            sample_valid_df,
            required_columns=["user_id", "score"],
            non_null_columns=["user_id"],
            unique_columns="user_id",
        )
        assert report["is_valid"] is True
        assert len(pipeline.execution_summary) == 1
        assert pipeline.execution_summary[0]["stage"] == "Validation"

    def test_dataprocessor_fluent_validation(self, sample_valid_df: pd.DataFrame) -> None:
        processor = DataProcessor()
        processor.set_data(sample_valid_df)
        null_res = processor.validate_null_constraints(columns=["user_id", "username"])
        assert all(null_res.values())
        unique_res = processor.validate_unique_constraints(columns="user_id")
        assert unique_res is True
        allowed_res = processor.validate_allowed_values(
            allowed_values={"status": ["active", "pending", "inactive"]}
        )
        assert allowed_res["status"] is True
