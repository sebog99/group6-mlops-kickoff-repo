"""
Module: Test Data Loader
------------------------
Tests for src/load_data.py
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from src.load_data import load_raw_data


def test_load_raw_data_returns_dataframe(tmp_path):
    """load_raw_data must return a DataFrame when file exists."""
    dummy_csv = tmp_path / "test_data.csv"
    dummy_csv.write_text("col1,col2\n1,a\n2,b\n")

    result = load_raw_data(dummy_csv)

    assert isinstance(result, pd.DataFrame)


def test_load_raw_data_correct_shape(tmp_path):
    """DataFrame must have the expected number of rows and columns."""
    dummy_csv = tmp_path / "test_data.csv"
    dummy_csv.write_text("col1,col2\n1,a\n2,b\n3,c\n")

    result = load_raw_data(dummy_csv)

    assert result.shape == (3, 2)


def test_load_raw_data_correct_columns(tmp_path):
    """DataFrame must preserve column names from the CSV."""
    dummy_csv = tmp_path / "test_data.csv"
    dummy_csv.write_text("col1,col2\n1,a\n2,b\n")

    result = load_raw_data(dummy_csv)

    assert list(result.columns) == ["col1", "col2"]


def test_load_raw_data_file_not_found():
    """load_raw_data must raise FileNotFoundError for missing files."""
    missing_path = Path("non/existent/file.csv")

    with pytest.raises(FileNotFoundError):
        load_raw_data(missing_path)