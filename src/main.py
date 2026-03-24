"""
Module: main.py
---------------
Role: Orchestrates the entire ML pipeline.

Pipeline steps:
1. Load raw data
2. Clean dataset
3. Validate schema
4. Feature engineering
5. Train/Test split
6. Model training
7. Model evaluation
8. Run inference and save predictions
"""

import os
import logging
from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

from src.load_data import load_raw_data
from src.clean_data import clean_data
from src.validate import validate_dataframe
from src.features import engineer_features, get_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference
from src.logger import configure_logging

logger = logging.getLogger(__name__)


# -----------------------------------------------------
# Config loader
# -----------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------
# Main Pipeline
# -----------------------------------------------------

def main() -> None:

    # Load configuration
    cfg = load_config()

    # Configure logging first
    configure_logging(
        log_level=cfg["logging"]["level"],
        log_file=Path(cfg["paths"]["log_file"]),
    )

    logger.info("Starting ML pipeline...")

    # Paths
    raw_path = Path(cfg["paths"]["raw_data"])
    clean_path = Path(cfg["paths"]["processed_data"])
    model_path = Path(cfg["paths"]["model_artifact"])
    preds_path = Path(cfg["paths"]["predictions_artifact"])

    # Validation config
    drop_na = cfg["validation"]["drop_na"]
    required_columns = cfg["validation"]["required_columns"]

    # Problem config
    target_column = cfg["problem"]["target_column"]
    problem_type = cfg["problem"]["problem_type"]
    identifier_column = cfg["problem"]["identifier_column"]

    # Split config
    test_size = cfg["split"]["test_size"]
    random_state = cfg["split"]["random_state"]

    # Model hyperparameters
    max_iter = cfg["training"]["classification"]["max_iter"]
    model_random_state = cfg["training"]["classification"]["random_state"]

    # Ensure folders exist
    for p in [clean_path, model_path, preds_path]:
        os.makedirs(p.parent, exist_ok=True)

    # -------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------

    logger.info("STEP 1: Loading raw data")
    df_raw = load_raw_data(raw_path)

    # -------------------------------------------------
    # STEP 2: Clean Data
    # -------------------------------------------------

    logger.info("STEP 2: Cleaning data")
    df_clean = clean_data(df_raw, drop_na=drop_na)

    df_clean.to_csv(clean_path, index=False)
    logger.info(f"Clean dataset saved to {clean_path}")

    # -------------------------------------------------
    # STEP 3: Validate Data
    # -------------------------------------------------

    logger.info("STEP 3: Validating dataset")
    validate_dataframe(df_clean, required_columns)

    # -------------------------------------------------
    # STEP 4: Feature Engineering
    # -------------------------------------------------

    logger.info("STEP 4: Feature engineering")

    df_eng = engineer_features(df_clean)

    X = df_eng.drop(columns=[target_column])
    y = df_eng[target_column].map({"No": 0, "Yes": 1})

    # -------------------------------------------------
    # STEP 5: Train/Test Split
    # -------------------------------------------------

    logger.info("STEP 5: Train/Test split")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # -------------------------------------------------
    # STEP 6: Preprocessor
    # -------------------------------------------------

    logger.info("STEP 6: Creating preprocessing pipeline")

    preprocessor = get_preprocessor(X_train)

    # -------------------------------------------------
    # STEP 7: Train Model
    # -------------------------------------------------

    logger.info("STEP 7: Training model")

    pipeline = train_model(
        X_train,
        y_train,
        preprocessor,
        model_path=str(model_path),
        max_iter=max_iter,
        random_state=model_random_state
    )

    # -------------------------------------------------
    # STEP 8: Evaluate Model
    # -------------------------------------------------

    logger.info("STEP 8: Evaluating model")

    metric_value = evaluate_model(
        pipeline,
        X_test,
        y_test,
        problem_type=problem_type
    )

    logger.info(f"Evaluation metric value: {metric_value:.4f}")

    # -------------------------------------------------
    # STEP 9: Inference
    # -------------------------------------------------

    logger.info("STEP 9: Running inference")

    predictions_df = run_inference(pipeline, X_test)

    predictions_df.to_csv(preds_path, index=False)

    logger.info(f"Predictions saved to {preds_path}")

    logger.info("Pipeline completed successfully!")


# -----------------------------------------------------
# Script Entry Point
# -----------------------------------------------------

if __name__ == "__main__":
    main()
