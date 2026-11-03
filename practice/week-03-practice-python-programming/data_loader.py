"""
data_loader.py

A modular, reusable Python data loading component providing both functional
and Object-Oriented Programming (OOP) interfaces for loading, cleansing,
summarizing, and serializing tabular datasets with strict type hints.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd  # type: ignore[import-untyped]

__all__ = [
    "BaseDataLoader",
    "DataLoader",
    "CSVDataLoader",
    "JSONDataLoader",
    "load_csv",
    "transform_data",
    "save_csv",
    "get_data_summary",
]


# ============================================================================
# Section 1: Functional Loading & Cleansing Utilities
# ============================================================================

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


# ============================================================================
# Section 2: Object-Oriented Data Loader Classes (OOP Hierarchy)
# ============================================================================

class BaseDataLoader(ABC):
    """
    Abstract Base Class establishing the contract for all DataLoader implementations.

    Provides common state management, path resolution, cleansing delegation,
    and metadata inspection routines. Subclasses must implement the abstract
    methods `load` and `save` for their specific file or data formats.
    """

    def __init__(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the base loader with an optional file path.

        Parameters
        ----------
        file_path : Optional[Union[str, Path]], default None
            Default file path bound to this loader instance.
        """
        self.file_path: Optional[Path] = (
            Path(file_path).expanduser().resolve()
            if file_path is not None and str(file_path).strip() != ""
            else None
        )
        self._data: Optional[pd.DataFrame] = None

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Return the currently loaded DataFrame, or None if no dataset is loaded."""
        return self._data

    @property
    def is_loaded(self) -> bool:
        """Return True if a DataFrame is currently loaded in memory."""
        return self._data is not None

    def _resolve_source_path(self, source: Optional[Union[str, Path]]) -> Path:
        """
        Resolve and validate the source file path.

        Parameters
        ----------
        source : Optional[Union[str, Path]]
            Explicit source file path, or None to fall back to `self.file_path`.

        Returns
        -------
        Path
            The resolved and validated file path.

        Raises
        ------
        ValueError
            If no path is provided or if the path points to a directory.
        FileNotFoundError
            If the resolved path does not exist.
        """
        if source is not None and str(source).strip() != "":
            target = Path(source).expanduser().resolve()
        elif self.file_path is not None:
            target = self.file_path
        else:
            raise ValueError("No file_path specified and loader instance has no bound file_path.")

        if not target.exists():
            raise FileNotFoundError(f"File not found at: {target}")
        if target.is_dir():
            raise ValueError(f"Expected a file path, but got a directory: {target}")
        return target

    def _resolve_dest_path(self, destination: Optional[Union[str, Path]]) -> Path:
        """
        Resolve and validate the destination file path.

        Parameters
        ----------
        destination : Optional[Union[str, Path]]
            Explicit destination file path, or None to fall back to `self.file_path`.

        Returns
        -------
        Path
            The resolved destination path.

        Raises
        ------
        ValueError
            If no destination path is provided or if it points to an existing directory.
        """
        if destination is not None and str(destination).strip() != "":
            target = Path(destination).expanduser().resolve()
        elif self.file_path is not None:
            target = self.file_path
        else:
            raise ValueError("No destination path specified and loader instance has no bound file_path.")

        if target.is_dir():
            raise ValueError(f"Expected a file path destination, but got a directory: {target}")
        return target

    @abstractmethod
    def load(self, source: Optional[Union[str, Path]] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Abstract method to load data into a pandas DataFrame.

        Parameters
        ----------
        source : Optional[Union[str, Path]], default None
            Target file path to read. Defaults to `self.file_path`.
        **kwargs : Any
            Format-specific loading arguments.

        Returns
        -------
        pd.DataFrame
            The loaded dataset.
        """
        pass

    @abstractmethod
    def save(
        self,
        df: Optional[pd.DataFrame] = None,
        destination: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Path:
        """
        Abstract method to persist a DataFrame.

        Parameters
        ----------
        df : Optional[pd.DataFrame], default None
            The DataFrame to save. If None, uses `self.data`.
        destination : Optional[Union[str, Path]], default None
            Output path. If None, uses `self.file_path`.
        **kwargs : Any
            Format-specific serialization arguments.

        Returns
        -------
        Path
            The path where data was written.
        """
        pass

    def clean(
        self,
        df: Optional[pd.DataFrame] = None,
        drop_na: bool = True,
        drop_duplicates: bool = False,
        strip_whitespace: bool = True,
        fill_na_value: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Apply data cleansing transformations to a DataFrame.

        If `df` is None, cleans the internally cached `self._data` and updates
        `self._data` with the cleaned result.

        Parameters
        ----------
        df : Optional[pd.DataFrame], default None
            DataFrame to clean. If None, uses `self._data`.
        drop_na : bool, default True
            Whether to drop rows with missing values.
        drop_duplicates : bool, default False
            Whether to drop duplicate rows.
        strip_whitespace : bool, default True
            Whether to trim whitespace on strings.
        fill_na_value : Optional[Any], default None
            Value to fill missing values if drop_na is False.

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame.

        Raises
        ------
        ValueError
            If no DataFrame is provided and no data is currently loaded.
        """
        target_df = df if df is not None else self._data
        if target_df is None:
            raise ValueError("No DataFrame provided and loader contains no loaded data to clean.")

        cleaned = transform_data(
            target_df,
            drop_na=drop_na,
            drop_duplicates=drop_duplicates,
            strip_whitespace=strip_whitespace,
            fill_na_value=fill_na_value,
        )
        if df is None:
            self._data = cleaned
        return cleaned

    def get_summary(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate structural summary metadata for a DataFrame.

        If `df` is None, inspects the internally cached `self._data`.

        Parameters
        ----------
        df : Optional[pd.DataFrame], default None
            DataFrame to summarize. If None, uses `self._data`.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing row count, column count, null counts, and dtypes.

        Raises
        ------
        ValueError
            If no DataFrame is provided and no data is currently loaded.
        """
        target_df = df if df is not None else self._data
        if target_df is None:
            raise ValueError("No DataFrame provided and loader contains no loaded data to summarize.")

        return get_data_summary(target_df)


class DataLoader(BaseDataLoader):
    """
    Object-Oriented CSV Data Loader inheriting from BaseDataLoader.

    Provides robust CSV ingestion, cleaning, metadata summary generation,
    and serialization with explicit type contracts.
    """

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
    ) -> None:
        """
        Initialize the CSV DataLoader.

        Parameters
        ----------
        file_path : Optional[Union[str, Path]], default None
            Default CSV file path.
        delimiter : str, default ","
            CSV field delimiter.
        encoding : str, default "utf-8"
            File encoding.
        """
        super().__init__(file_path=file_path)
        self.delimiter: str = delimiter
        self.encoding: str = encoding

    def load(
        self,
        source: Optional[Union[str, Path]] = None,
        delimiter: Optional[str] = None,
        encoding: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame and cache it in `self.data`.

        Parameters
        ----------
        source : Optional[Union[str, Path]], default None
            Source CSV file path. Defaults to `self.file_path`.
        delimiter : Optional[str], default None
            Field delimiter. Defaults to `self.delimiter`.
        encoding : Optional[str], default None
            File encoding. Defaults to `self.encoding`.
        **kwargs : Any
            Additional options passed to `pd.read_csv`.

        Returns
        -------
        pd.DataFrame
            The loaded DataFrame.
        """
        target_path = self._resolve_source_path(source)
        sep = delimiter if delimiter is not None else self.delimiter
        enc = encoding if encoding is not None else self.encoding

        df = load_csv(target_path, delimiter=sep, encoding=enc, **kwargs)
        self._data = df
        return df

    def save(
        self,
        df: Optional[pd.DataFrame] = None,
        destination: Optional[Union[str, Path]] = None,
        index: bool = False,
        encoding: Optional[str] = None,
        **kwargs: Any,
    ) -> Path:
        """
        Save a DataFrame to a CSV destination file.

        Parameters
        ----------
        df : Optional[pd.DataFrame], default None
            DataFrame to persist. If None, uses `self.data`.
        destination : Optional[Union[str, Path]], default None
            Destination path. If None, uses `self.file_path`.
        index : bool, default False
            Whether to write row names (index).
        encoding : Optional[str], default None
            Encoding to use. Defaults to `self.encoding`.
        **kwargs : Any
            Additional arguments passed to `pd.DataFrame.to_csv`.

        Returns
        -------
        Path
            The resolved destination path.

        Raises
        ------
        ValueError
            If no DataFrame is provided and no data is loaded.
        """
        target_df = df if df is not None else self._data
        if target_df is None:
            raise ValueError("No DataFrame provided or currently loaded to save.")

        dest_path = self._resolve_dest_path(destination)
        enc = encoding if encoding is not None else self.encoding
        return save_csv(target_df, dest_path, index=index, encoding=enc, **kwargs)


CSVDataLoader = DataLoader


class JSONDataLoader(BaseDataLoader):
    """
    Object-Oriented JSON Data Loader inheriting from BaseDataLoader.

    Demonstrates class inheritance and polymorphic data loading across
    different serialization formats adhering to the same base contracts.
    """

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        orient: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> None:
        """
        Initialize the JSON DataLoader.

        Parameters
        ----------
        file_path : Optional[Union[str, Path]], default None
            Default JSON file path.
        orient : Optional[str], default None
            Expected JSON format ('records', 'split', 'index', etc.).
        encoding : str, default "utf-8"
            File encoding.
        """
        super().__init__(file_path=file_path)
        self.orient: Optional[str] = orient
        self.encoding: str = encoding

    def load(
        self,
        source: Optional[Union[str, Path]] = None,
        orient: Optional[str] = None,
        encoding: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load a JSON file into a pandas DataFrame and cache it.

        Parameters
        ----------
        source : Optional[Union[str, Path]], default None
            Target JSON file path. Defaults to `self.file_path`.
        orient : Optional[str], default None
            JSON orientation. Defaults to `self.orient`.
        encoding : Optional[str], default None
            File encoding. Defaults to `self.encoding`.
        **kwargs : Any
            Additional arguments passed to `pd.read_json`.

        Returns
        -------
        pd.DataFrame
            The loaded DataFrame.
        """
        target_path = self._resolve_source_path(source)
        target_orient = orient if orient is not None else self.orient
        target_encoding = encoding if encoding is not None else self.encoding

        df: pd.DataFrame = pd.read_json(
            target_path,
            orient=target_orient,
            encoding=target_encoding,
            **kwargs,
        )
        self._data = df
        return df

    def save(
        self,
        df: Optional[pd.DataFrame] = None,
        destination: Optional[Union[str, Path]] = None,
        orient: Optional[str] = None,
        indent: int = 2,
        **kwargs: Any,
    ) -> Path:
        """
        Save a DataFrame to a JSON file.

        Parameters
        ----------
        df : Optional[pd.DataFrame], default None
            DataFrame to persist. Defaults to `self.data`.
        destination : Optional[Union[str, Path]], default None
            Destination JSON file path. Defaults to `self.file_path`.
        orient : Optional[str], default None
            JSON format orientation (e.g. 'records'). Defaults to `self.orient` or 'records'.
        indent : int, default 2
            Number of indentation spaces.
        **kwargs : Any
            Additional arguments passed to `pd.DataFrame.to_json`.

        Returns
        -------
        Path
            The resolved destination path.

        Raises
        ------
        ValueError
            If no DataFrame is provided and no data is loaded.
        """
        target_df = df if df is not None else self._data
        if target_df is None:
            raise ValueError("No DataFrame provided or currently loaded to save.")

        if not isinstance(target_df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(target_df).__name__}")

        dest_path = self._resolve_dest_path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        target_orient = orient if orient is not None else (self.orient or "records")

        target_df.to_json(
            dest_path,
            orient=target_orient,
            indent=indent,
            **kwargs,
        )
        return dest_path


if __name__ == "__main__":
    print("data_loader module loaded successfully.")
