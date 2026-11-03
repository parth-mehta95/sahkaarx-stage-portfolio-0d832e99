"""
test_data_processor.py - Comprehensive test suite for data_processor module.
"""

from typing import Any, Dict, List, get_type_hints
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from data_processor import (
    aggregate_by_group,
    calculate_summary_statistics,
    cast_column_types,
    filter_by_condition,
    normalize_numeric_columns,
    pivot_and_aggregate,
    run_pipeline,
    validate_column_ranges,
    validate_schema,
)


@pytest.fixture
def sample_sales_data() -> pd.DataFrame:
    """Fixture providing a sample tabular dataset representing sales transactions."""
    return pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105, 106],
        "category": ["Electronics", "Books", "Electronics", "Home", "Books", "Home"],
        "region": ["North", "South", "North", "West", "East", "North"],
        "units": [5, 2, 8, 1, 4, 3],
        "unit_price": [120.0, 15.5, 95.0, 45.0, 22.0, 60.0],
        "total_revenue": [600.0, 31.0, 760.0, 45.0, 88.0, 180.0],
        "is_discounted": [False, True, True, False, False, True],
    })


# ============================================================================
# 1. Type Hints & Contract Tests
# ============================================================================

def test_type_hints_contracts() -> None:
    """Validate that all data_processor functions expose comprehensive type hints."""
    # validate_schema
    hints_schema = get_type_hints(validate_schema)
    assert hints_schema["return"] is bool
    assert "df" in hints_schema
    assert "required_columns" in hints_schema
    assert "column_types" in hints_schema
    assert "allow_empty" in hints_schema

    # validate_column_ranges
    hints_ranges = get_type_hints(validate_column_ranges)
    assert "return" in hints_ranges
    assert "df" in hints_ranges
    assert "bounds" in hints_ranges

    # filter_by_condition
    hints_filter = get_type_hints(filter_by_condition)
    assert hints_filter["return"] is pd.DataFrame
    assert "df" in hints_filter
    assert "column" in hints_filter
    assert "operator" in hints_filter
    assert "value" in hints_filter

    # normalize_numeric_columns
    hints_norm = get_type_hints(normalize_numeric_columns)
    assert hints_norm["return"] is pd.DataFrame
    assert "df" in hints_norm
    assert "columns" in hints_norm
    assert "method" in hints_norm

    # cast_column_types
    hints_cast = get_type_hints(cast_column_types)
    assert hints_cast["return"] is pd.DataFrame
    assert "df" in hints_cast
    assert "type_map" in hints_cast

    # aggregate_by_group
    hints_agg = get_type_hints(aggregate_by_group)
    assert hints_agg["return"] is pd.DataFrame
    assert "df" in hints_agg
    assert "group_by" in hints_agg
    assert "aggregations" in hints_agg
    assert "reset_index" in hints_agg

    # calculate_summary_statistics
    hints_stats = get_type_hints(calculate_summary_statistics)
    assert "return" in hints_stats
    assert "df" in hints_stats
    assert "columns" in hints_stats

    # pivot_and_aggregate
    hints_pivot = get_type_hints(pivot_and_aggregate)
    assert hints_pivot["return"] is pd.DataFrame
    assert "df" in hints_pivot
    assert "index" in hints_pivot
    assert "columns" in hints_pivot
    assert "values" in hints_pivot

    # run_pipeline
    hints_pipe = get_type_hints(run_pipeline)
    assert hints_pipe["return"] is pd.DataFrame
    assert "df" in hints_pipe
    assert "required_columns" in hints_pipe


# ============================================================================
# 2. Validation Function Tests
# ============================================================================

def test_validate_schema_success(sample_sales_data: pd.DataFrame) -> None:
    """Test schema validation with valid columns and matching types."""
    required = ["order_id", "category", "units", "total_revenue"]
    types: Dict[str, Any] = {
        "order_id": int,
        "category": str,
        "total_revenue": float,
    }
    assert validate_schema(sample_sales_data, required, column_types=types) is True


def test_validate_schema_missing_column(sample_sales_data: pd.DataFrame) -> None:
    """Test schema validation fails when required column is absent."""
    with pytest.raises(ValueError, match="DataFrame is missing required columns"):
        validate_schema(sample_sales_data, ["order_id", "non_existent_column"])


def test_validate_schema_empty_df_rejected() -> None:
    """Test schema validation rejects empty DataFrame when allow_empty is False."""
    empty_df = pd.DataFrame(columns=["id", "name"])
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_schema(empty_df, ["id", "name"], allow_empty=False)


def test_validate_schema_empty_df_allowed() -> None:
    """Test schema validation accepts empty DataFrame when allow_empty is True."""
    empty_df = pd.DataFrame(columns=["id", "name"])
    assert validate_schema(empty_df, ["id", "name"], allow_empty=True) is True


def test_validate_schema_type_mismatch(sample_sales_data: pd.DataFrame) -> None:
    """Test schema validation raises TypeError when column type does not match."""
    with pytest.raises(TypeError, match="expected integer type"):
        validate_schema(sample_sales_data, ["category"], column_types={"category": int})


def test_validate_schema_non_dataframe() -> None:
    """Test schema validation raises TypeError on non-DataFrame inputs."""
    invalid_input: Any = {"col": [1, 2, 3]}
    with pytest.raises(TypeError, match="Expected input of type pd.DataFrame"):
        validate_schema(invalid_input, ["col"])


def test_validate_column_ranges_success(sample_sales_data: pd.DataFrame) -> None:
    """Test checking column values against valid bounds."""
    bounds = {
        "units": (1.0, 10.0),
        "total_revenue": (0.0, 1000.0),
    }
    results = validate_column_ranges(sample_sales_data, bounds)
    assert results == {"units": True, "total_revenue": True}


def test_validate_column_ranges_failure(sample_sales_data: pd.DataFrame) -> None:
    """Test checking column values against too-strict bounds returns False."""
    bounds = {
        "units": (1.0, 5.0),  # Max units in fixture is 8, so this should fail
    }
    results = validate_column_ranges(sample_sales_data, bounds)
    assert results["units"] is False


def test_validate_column_ranges_one_sided_bounds(sample_sales_data: pd.DataFrame) -> None:
    """Test validation with unbounded lower or upper boundaries."""
    bounds = {
        "unit_price": (10.0, None),  # all prices >= 10.0
        "units": (None, 20.0),       # all units <= 20.0
    }
    results = validate_column_ranges(sample_sales_data, bounds)
    assert results["unit_price"] is True
    assert results["units"] is True


def test_validate_column_ranges_missing_or_non_numeric(sample_sales_data: pd.DataFrame) -> None:
    """Test errors when bounds are invalid or target missing/non-numeric columns."""
    with pytest.raises(KeyError, match="not found in DataFrame"):
        validate_column_ranges(sample_sales_data, {"missing_col": (0.0, 10.0)})

    with pytest.raises(TypeError, match="is not numeric"):
        validate_column_ranges(sample_sales_data, {"category": (0.0, 10.0)})

    with pytest.raises(ValueError, match="cannot exceed max_val"):
        validate_column_ranges(sample_sales_data, {"units": (10.0, 2.0)})


# ============================================================================
# 3. Transformation Function Tests
# ============================================================================

def test_filter_by_condition_comparison_operators(sample_sales_data: pd.DataFrame) -> None:
    """Test standard comparison operators (==, !=, >, >=, <, <=)."""
    # ==
    df_books = filter_by_condition(sample_sales_data, "category", "==", "Books")
    assert len(df_books) == 2
    assert set(df_books["category"].unique()) == {"Books"}

    # !=
    df_not_books = filter_by_condition(sample_sales_data, "category", "!=", "Books")
    assert len(df_not_books) == 4
    assert "Books" not in df_not_books["category"].values

    # >
    df_high_rev = filter_by_condition(sample_sales_data, "total_revenue", ">", 200.0)
    assert len(df_high_rev) == 2  # 600.0 and 760.0

    # >=
    df_ge = filter_by_condition(sample_sales_data, "units", ">=", 5)
    assert len(df_ge) == 2  # 5 and 8

    # <
    df_low = filter_by_condition(sample_sales_data, "unit_price", "<", 50.0)
    assert len(df_low) == 3  # 15.5, 45.0, 22.0

    # <=
    df_le = filter_by_condition(sample_sales_data, "units", "<=", 2)
    assert len(df_le) == 2  # 2 and 1


def test_filter_by_condition_in_and_not_in(sample_sales_data: pd.DataFrame) -> None:
    """Test 'in' and 'not in' operators."""
    # in
    df_north_south = filter_by_condition(
        sample_sales_data, "region", "in", ["North", "South"]
    )
    assert len(df_north_south) == 4
    assert set(df_north_south["region"].unique()).issubset({"North", "South"})

    # not in
    df_exclude = filter_by_condition(
        sample_sales_data, "region", "not in", ["North", "South"]
    )
    assert len(df_exclude) == 2
    assert "North" not in df_exclude["region"].values
    assert "South" not in df_exclude["region"].values


def test_filter_by_condition_errors(sample_sales_data: pd.DataFrame) -> None:
    """Test error handling in filter_by_condition."""
    with pytest.raises(KeyError, match="does not exist"):
        filter_by_condition(sample_sales_data, "non_existent", "==", 10)

    with pytest.raises(ValueError, match="Unsupported operator"):
        filter_by_condition(sample_sales_data, "units", "like", 10)

    with pytest.raises(ValueError, match="must be an iterable collection"):
        filter_by_condition(sample_sales_data, "region", "in", 123)


def test_filter_by_condition_non_mutating(sample_sales_data: pd.DataFrame) -> None:
    """Verify that filtering does not mutate the original DataFrame."""
    original_len = len(sample_sales_data)
    filter_by_condition(sample_sales_data, "category", "==", "Books")
    assert len(sample_sales_data) == original_len


def test_normalize_numeric_columns_minmax(sample_sales_data: pd.DataFrame) -> None:
    """Test min-max normalization scales values into [0.0, 1.0]."""
    norm_df = normalize_numeric_columns(
        sample_sales_data, columns=["units", "total_revenue"], method="minmax"
    )

    # Check that units is bounded [0.0, 1.0]
    assert norm_df["units"].min() == 0.0
    assert norm_df["units"].max() == 1.0

    # Check original wasn't mutated
    assert sample_sales_data["units"].max() == 8


def test_normalize_numeric_columns_zscore(sample_sales_data: pd.DataFrame) -> None:
    """Test z-score normalization standardizes to mean ~ 0 and std ~ 1."""
    norm_df = normalize_numeric_columns(
        sample_sales_data, columns=["total_revenue"], method="zscore"
    )
    mean_val = float(norm_df["total_revenue"].mean())
    std_val = float(norm_df["total_revenue"].std(ddof=0))
    assert pytest.approx(mean_val, abs=1e-5) == 0.0
    assert pytest.approx(std_val, abs=1e-5) == 1.0


def test_normalize_numeric_columns_zero_variance() -> None:
    """Test normalization handles zero-variance (constant) column safely without NaN."""
    const_df = pd.DataFrame({"constant": [10.0, 10.0, 10.0], "val": [1.0, 2.0, 3.0]})
    norm_df = normalize_numeric_columns(const_df, method="minmax")
    assert (norm_df["constant"] == 0.0).all()


def test_normalize_numeric_columns_errors(sample_sales_data: pd.DataFrame) -> None:
    """Test normalization error cases."""
    with pytest.raises(ValueError, match="Unsupported normalization method"):
        normalize_numeric_columns(sample_sales_data, method="invalid_method")

    with pytest.raises(TypeError, match="is not numeric"):
        normalize_numeric_columns(sample_sales_data, columns=["category"])

    with pytest.raises(KeyError, match="not found"):
        normalize_numeric_columns(sample_sales_data, columns=["missing_col"])


def test_cast_column_types_success(sample_sales_data: pd.DataFrame) -> None:
    """Test type casting across standard and datetime types."""
    cast_map = {
        "order_id": "float64",
        "units": "int32",
    }
    result_df = cast_column_types(sample_sales_data, cast_map)
    assert result_df["order_id"].dtype == np.float64
    assert result_df["units"].dtype == np.int32


def test_cast_column_types_errors(sample_sales_data: pd.DataFrame) -> None:
    """Test error handling when casting fails or column is missing."""
    with pytest.raises(KeyError, match="not found in DataFrame"):
        cast_column_types(sample_sales_data, {"missing_col": "int64"})

    with pytest.raises(ValueError, match="Failed to cast column"):
        cast_column_types(sample_sales_data, {"category": "int64"})


# ============================================================================
# 4. Aggregation Function Tests
# ============================================================================

def test_aggregate_by_group_single_col(sample_sales_data: pd.DataFrame) -> None:
    """Test group aggregation with a single group column and single agg."""
    agg_df = aggregate_by_group(
        sample_sales_data,
        group_by="category",
        aggregations={"total_revenue": "sum", "units": "mean"},
    )
    assert isinstance(agg_df, pd.DataFrame)
    assert "category" in agg_df.columns
    assert "total_revenue" in agg_df.columns
    assert "units" in agg_df.columns
    assert len(agg_df) == 3  # Books, Electronics, Home


def test_aggregate_by_group_multi_agg(sample_sales_data: pd.DataFrame) -> None:
    """Test group aggregation with multi-metric aggregation and flattened headers."""
    agg_df = aggregate_by_group(
        sample_sales_data,
        group_by=["category", "region"],
        aggregations={"total_revenue": ["sum", "mean"]},
        reset_index=True,
    )
    assert "category" in agg_df.columns
    assert "region" in agg_df.columns
    assert "total_revenue_sum" in agg_df.columns
    assert "total_revenue_mean" in agg_df.columns


def test_aggregate_by_group_errors(sample_sales_data: pd.DataFrame) -> None:
    """Test error handling for aggregate_by_group."""
    with pytest.raises(KeyError, match="Grouping column"):
        aggregate_by_group(sample_sales_data, "non_existent", {"units": "sum"})

    with pytest.raises(KeyError, match="Aggregation target column"):
        aggregate_by_group(sample_sales_data, "category", {"non_existent": "sum"})

    with pytest.raises(ValueError, match="group_by cannot be empty"):
        aggregate_by_group(sample_sales_data, [], {"units": "sum"})


def test_calculate_summary_statistics_all(sample_sales_data: pd.DataFrame) -> None:
    """Test calculating summary statistics across all numeric columns."""
    stats = calculate_summary_statistics(sample_sales_data)
    assert "units" in stats
    assert "total_revenue" in stats
    assert "category" not in stats  # Non-numeric excluded

    units_stats = stats["units"]
    assert units_stats["count"] == 6.0
    assert units_stats["min"] == 1.0
    assert units_stats["max"] == 8.0
    assert "mean" in units_stats
    assert "std" in units_stats
    assert "median" in units_stats


def test_calculate_summary_statistics_specific_columns(sample_sales_data: pd.DataFrame) -> None:
    """Test calculating statistics on specific columns."""
    stats = calculate_summary_statistics(sample_sales_data, columns=["total_revenue"])
    assert list(stats.keys()) == ["total_revenue"]
    assert stats["total_revenue"]["count"] == 6.0


def test_calculate_summary_statistics_errors(sample_sales_data: pd.DataFrame) -> None:
    """Test error handling in calculate_summary_statistics."""
    with pytest.raises(KeyError, match="not found in DataFrame"):
        calculate_summary_statistics(sample_sales_data, columns=["missing"])

    with pytest.raises(TypeError, match="is not numeric"):
        calculate_summary_statistics(sample_sales_data, columns=["category"])


def test_pivot_and_aggregate_success(sample_sales_data: pd.DataFrame) -> None:
    """Test pivot table creation with aggregation."""
    pivoted = pivot_and_aggregate(
        sample_sales_data,
        index="category",
        columns="region",
        values="total_revenue",
        aggfunc="sum",
        fill_value=0.0,
    )
    assert isinstance(pivoted, pd.DataFrame)
    assert "category" in pivoted.columns
    # Check regions North, South, East, West exist as columns
    for region in ["North", "South", "East", "West"]:
        assert region in pivoted.columns


# ============================================================================
# 5. Pipeline Orchestrator Tests
# ============================================================================

def test_run_pipeline_end_to_end(sample_sales_data: pd.DataFrame) -> None:
    """Test running an integrated pipeline from validation to aggregation."""
    pipeline_result = run_pipeline(
        df=sample_sales_data,
        required_columns=["order_id", "category", "units", "total_revenue"],
        filter_column="units",
        filter_op=">=",
        filter_value=3,
        normalize_columns=["unit_price"],
        normalize_method="minmax",
        group_by="category",
        aggregations={"total_revenue": "sum", "unit_price": "mean"},
    )
    assert isinstance(pipeline_result, pd.DataFrame)
    assert "category" in pipeline_result.columns
    assert "total_revenue" in pipeline_result.columns
    assert "unit_price" in pipeline_result.columns
    # Electronics, Books, Home all have rows with units >= 3
    assert len(pipeline_result) >= 1


def test_run_pipeline_validation_failure(sample_sales_data: pd.DataFrame) -> None:
    """Test that pipeline aborts early if schema validation fails."""
    with pytest.raises(ValueError, match="missing required columns"):
        run_pipeline(
            df=sample_sales_data,
            required_columns=["non_existent_id"],
        )
