"""
test_data_loader.py - Comprehensive test suite for data_loader module.
"""

from pathlib import Path
from typing import get_type_hints
import pandas as pd
import pytest

from data_loader import get_data_summary, load_csv, save_csv, transform_data


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Fixture providing a sample CSV file with various data characteristics."""
    csv_file = tmp_path / "sample.csv"
    csv_content = (
        " name , age , salary \n"
        " Alice , 28 , 75000 \n"
        " Bob ,  , 60000 \n"
        " Charlie , 35 , 90000 \n"
        " Alice , 28 , 75000 \n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")
    return csv_file


def test_type_hints_contract() -> None:
    """Validate that load_csv and transform_data have explicit type hints."""
    load_hints = get_type_hints(load_csv)
    assert "return" in load_hints
    assert "file_path" in load_hints
    assert "delimiter" in load_hints
    assert "encoding" in load_hints

    transform_hints = get_type_hints(transform_data)
    assert "return" in transform_hints
    assert "df" in transform_hints
    assert "drop_na" in transform_hints
    assert "drop_duplicates" in transform_hints
    assert "strip_whitespace" in transform_hints


def test_load_csv_success(sample_csv: Path) -> None:
    """Test loading a valid CSV file using both str and Path."""
    df_from_str = load_csv(str(sample_csv))
    assert isinstance(df_from_str, pd.DataFrame)
    assert len(df_from_str) == 4

    df_from_path = load_csv(sample_csv)
    assert isinstance(df_from_path, pd.DataFrame)
    assert df_from_path.shape == (4, 3)


def test_load_csv_file_not_found(tmp_path: Path) -> None:
    """Test that load_csv raises FileNotFoundError for missing files."""
    missing_file = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_csv(missing_file)


def test_load_csv_empty_path() -> None:
    """Test that load_csv raises ValueError when path is empty."""
    with pytest.raises(ValueError):
        load_csv("")


def test_load_csv_directory(tmp_path: Path) -> None:
    """Test that load_csv raises ValueError when a directory is passed."""
    with pytest.raises(ValueError):
        load_csv(tmp_path)


def test_transform_data_defaults(sample_csv: Path) -> None:
    """Test transform_data with default settings (drop_na=True, strip_whitespace=True)."""
    raw_df = load_csv(sample_csv)
    cleaned_df = transform_data(raw_df)

    # Column whitespace stripped
    assert list(cleaned_df.columns) == ["name", "age", "salary"]

    # Bob had missing age, so should be dropped
    assert "Bob" not in cleaned_df["name"].values

    # Whitespace stripped from values
    assert all(not val.startswith(" ") and not val.endswith(" ") for val in cleaned_df["name"])

    # Duplicate row (Alice) kept since drop_duplicates defaults to False
    assert len(cleaned_df[cleaned_df["name"] == "Alice"]) == 2


def test_transform_data_drop_duplicates(sample_csv: Path) -> None:
    """Test transform_data with drop_duplicates=True."""
    raw_df = load_csv(sample_csv)
    cleaned_df = transform_data(raw_df, drop_na=True, drop_duplicates=True)

    # Bob dropped (NA), duplicate Alice dropped
    assert len(cleaned_df) == 2
    assert list(cleaned_df["name"]) == ["Alice", "Charlie"]


def test_transform_data_keep_na(sample_csv: Path) -> None:
    """Test transform_data with drop_na=False."""
    raw_df = load_csv(sample_csv)
    transformed_df = transform_data(raw_df, drop_na=False)

    assert len(transformed_df) == 4
    assert "Bob" in transformed_df["name"].values


def test_transform_data_fill_na(sample_csv: Path) -> None:
    """Test transform_data with fill_na_value."""
    raw_df = load_csv(sample_csv)
    transformed_df = transform_data(raw_df, drop_na=False, fill_na_value="0")

    assert transformed_df.isna().sum().sum() == 0


def test_transform_data_invalid_type() -> None:
    """Test that transform_data raises TypeError when passed non-DataFrame."""
    with pytest.raises(TypeError):
        transform_data({"key": "value"})  # type: ignore[arg-type]


def test_save_csv(tmp_path: Path) -> None:
    """Test saving DataFrame to CSV."""
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    out_file = tmp_path / "output" / "saved.csv"
    saved_path = save_csv(df, out_file)

    assert saved_path.exists()
    reloaded = pd.read_csv(saved_path)
    assert len(reloaded) == 2


def test_get_data_summary() -> None:
    """Test generating dataset structural summary."""
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    summary = get_data_summary(df)

    assert summary["num_rows"] == 3
    assert summary["num_columns"] == 2
    assert summary["columns"] == ["a", "b"]
    assert summary["null_counts"]["a"] == 1
    assert summary["null_counts"]["b"] == 0

