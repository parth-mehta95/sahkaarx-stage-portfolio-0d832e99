"""
data_loader.py

A reusable Python module for loading CSV files and transforming data
with strict, clear type contracts and explicit type hints.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd  # type: ignore[import-untyped]


def load_csv(
    file_path: Union[str, Path],
    delimiter: str = ",",
    encoding: str = "utf-8",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with validation and clear type contracts.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the CSV file (either a string or a pathlib.Path object).
    delimiter : str, default ","
        The field delimiter used in the CSV file.
    encoding : str, default "utf-8"
        Encoding to use when reading the file.
    **kwargs : Any
        Additional keyword arguments passed to pandas.read_csv.

    Returns
    -------
    pd.DataFrame
        The loaded dataset as a pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the specified path.
    ValueError
        If file_path is empty or points to a directory.
    """
    if not file_path:
        raise ValueError("file_path cannot be empty.")

    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at: {path}")

    if path.is_dir():
        raise ValueError(f"Expected a file path, but got a directory: {path}")

    df: pd.DataFrame = pd.read_csv(
        path,
        delimiter=delimiter,
        encoding=encoding,
        **kwargs,
    )
    return df


def transform_data(
    df: pd.DataFrame,
    drop_na: bool = True,
    drop_duplicates: bool = False,
    strip_whitespace: bool = True,
    fill_na_value: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Apply data cleansing and transformations to a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to transform.
    drop_na : bool, default True
        Whether to drop rows containing any missing (NaN/null) values.
    drop_duplicates : bool, default False
        Whether to drop duplicate rows from the dataset.
    strip_whitespace : bool, default True
        Whether to strip leading/trailing whitespace from string columns and column names.
    fill_na_value : Optional[Any], default None
        If provided and drop_na is False, fills missing values with this value.

    Returns
    -------
    pd.DataFrame
        The cleaned and transformed DataFrame with an updated index.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected input of type pd.DataFrame, got {type(df).__name__}")

    # Work on a defensive copy to avoid mutating the original DataFrame
    transformed_df: pd.DataFrame = df.copy()

    # Clean whitespace in column headers
    if strip_whitespace:
        transformed_df.columns = pd.Index([
            col.strip() if isinstance(col, str) else col
            for col in transformed_df.columns
        ])

        # Strip whitespace from all string/object cells
        str_cols = transformed_df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            transformed_df[col] = transformed_df[col].apply(
                lambda val: val.strip() if isinstance(val, str) else val
            )
            # Treat empty strings as missing/null values
            transformed_df[col] = transformed_df[col].replace("", pd.NA)

    # Handle missing values
    if drop_na:
        transformed_df = transformed_df.dropna()
    elif fill_na_value is not None:
        transformed_df = transformed_df.fillna(fill_na_value)

    # Handle duplicates
    if drop_duplicates:
        transformed_df = transformed_df.drop_duplicates()

    # Reset index to maintain consecutive row indexing
    transformed_df = transformed_df.reset_index(drop=True)

    return transformed_df


def save_csv(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    index: bool = False,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> Path:
    """
    Save a pandas DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    file_path : Union[str, Path]
        The destination file path.
    index : bool, default False
        Whether to write row names (index).
    encoding : str, default "utf-8"
        The character encoding to use.
    **kwargs : Any
        Additional keyword arguments passed to DataFrame.to_csv.

    Returns
    -------
    Path
        The resolved Path where the file was saved.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    output_path = Path(file_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=index, encoding=encoding, **kwargs)
    return output_path


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a structural summary of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing shape, column names, null counts, and dtypes.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    return {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "null_counts": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
        "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
    }


if __name__ == "__main__":
    print("data_loader module loaded successfully.")
