import pytest
import pandas as pd
from unittest.mock import patch
from src.load_data import load_raw_data


@patch("pandas.read_csv")
def test_load_raw_data(mock_read_csv):
    """Verifica que los datos se carguen pasando la ruta correcta a pandas."""
    mock_read_csv.return_value = pd.DataFrame({"col1": [1, 2, 3]})
    df = load_raw_data("dummy/path.csv")

    assert not df.empty
    assert "col1" in df.columns
    mock_read_csv.assert_called_once_with("dummy/path.csv")
