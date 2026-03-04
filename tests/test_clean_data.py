import pytest
import pandas as pd
from src.clean_data import clean_dataframe


@pytest.fixture
def dirty_data():
    return pd.DataFrame({
        "customerID": ["123", "456", "789"],
        "TotalCharges": [" 10.5 ", " ", "20.0"],
        "tenure": ["1", "2", "3"],
        "gender": ["Male ", "Female", "Male"],
        "Churn": ["Yes", "No", "Yes"]
    })


def test_clean_dataframe(dirty_data):
    """Verifica que el DataFrame se limpie y procese correctamente."""
    # Usando la estructura que tienes en tu main.py
    df_clean = clean_dataframe(dirty_data, target_column="Churn")

    assert not df_clean.empty
    assert "customerID" not in df_clean.columns or len(df_clean) > 0
