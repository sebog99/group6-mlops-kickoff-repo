"""
Module: features.py
-------------------
Role: Create, select, and define the transformation logic for engineered features.
Input: Cleaned pandas.DataFrame.
Output: A scikit-learn ColumnTransformer object (the "recipe") and the engineered DataFrame.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Configuration of logging [cite: 14, 38]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def engineer_features(df: pd.DataFrame):
    """
    Apply domain-specific feature engineering.

    Why: Transforms raw data into a format that highlights underlying patterns 
    for the Logistic Regression model.
    """
    try:
        logging.info("Starting feature engineering...")
        df_eng = df.copy()

        # 1. Feature Selection: Drop non-predictive identifiers
        if 'customerID' in df_eng.columns:
            df_eng = df_eng.drop(columns=['customerID'])
            logging.info("Dropped 'customerID'.")

        # 2. Specific Logic: Ensure TotalCharges is numeric (if not handled in cleaner)
        # Why: TotalCharges often arrives as 'object' due to empty strings
        if 'TotalCharges' in df_eng.columns:
            df_eng['TotalCharges'] = pd.to_numeric(
                df_eng['TotalCharges'], errors='coerce')
            df_eng['TotalCharges'] = df_eng['TotalCharges'].fillna(0)

        logging.info("Feature engineering completed.")
        return df_eng

    except Exception as e:
        logging.error(f"Error in engineering features: {e}")
        raise e


def get_preprocessor(X: pd.DataFrame):
    """
    Defines the ColumnTransformer (The 'Recipe').

    Why Industry Grade? 
    Using a ColumnTransformer instead of pd.get_dummies prevents 'data leakage' 
    and ensures consistency between training and inference phases.
    """
    try:
        logging.info("Defining the preprocessor (ColumnTransformer)...")

        # Identify column types based on your notebook experiment
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_features = [
            col for col in X.columns
            if col not in numeric_features and X[col].dtype == 'object'
        ]

        # Numeric: Scaling is essential for Logistic Regression convergence
        numeric_transformer = StandardScaler()

        # Categorical: One-Hot Encoding for non-numeric features
        categorical_transformer = OneHotEncoder(
            handle_unknown='ignore', drop='first')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        logging.info(
            f"Preprocessor defined with {len(numeric_features)} numeric and {len(categorical_features)} categorical steps.")
        return preprocessor

    except Exception as e:
        logging.error(f"Error defining preprocessor: {e}")
        raise e
