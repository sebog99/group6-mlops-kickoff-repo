import pytest
import pandas as pd
from src.validate import validate_dataframe


def test_validate_dataframe_success():
    """Verifica que un DataFrame correcto pase la validación."""
    df = pd.DataFrame({
        "MonthlyCharges": [10.0, 20.0],
        "tenure": [5, 10],
        "SeniorCitizen": [0, 1],
        "Churn": ["No", "Yes"]
    })
    assert validate_dataframe(df, required_columns=["Churn"]) == True


def test_validate_dataframe_empty():
    """Verifica que un DataFrame vacío lance ValueError."""
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=[])
