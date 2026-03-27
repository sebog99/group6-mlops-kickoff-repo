import os
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import wandb
from dotenv import load_dotenv

from .clean_data import clean_data
from .validate import validate_dataframe
from .features import engineer_features
from .infer import run_inference

load_dotenv(override=False)

logger = logging.getLogger(__name__)

app = FastAPI(title="Churn API")


def load_model():
    """
    Load model from W&B prod artifact if WANDB_API_KEY is set,
    otherwise fall back to local models/model.joblib.
    """
    api_key = os.getenv("WANDB_API_KEY")
    entity = os.getenv("WANDB_ENTITY", "group6-mlops-kickoff-repo")
    project = os.getenv("WANDB_PROJECT", "telco-churn")

    if api_key:
        try:
            logger.info("Downloading model from W&B (alias: prod)...")
            api = wandb.Api()
            artifact = api.artifact(f"{entity}/{project}/telco_churn_model:prod")
            artifact_dir = artifact.download(root="models/")
            model_path = Path(artifact_dir) / "model.joblib"
            logger.info(f"Model downloaded to {model_path}")
            return joblib.load(model_path)
        except Exception as e:
            logger.warning(f"W&B download failed: {e}. Falling back to local model.")

    local_path = "models/model.joblib"
    logger.info(f"Loading model from local path: {local_path}")
    return joblib.load(local_path)


model = load_model()


class Customer(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str


class RequestData(BaseModel):
    customers: List[Customer]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: RequestData):
    try:
        df = pd.DataFrame([c.model_dump() for c in data.customers])

        df_clean = clean_data(df, drop_na=False)

        # Add dummy target BEFORE validation
        df_clean["Churn"] = "No"

        validate_dataframe(df_clean, df_clean.columns.tolist())

        ids = df_clean["customerID"].copy()

        df_features = engineer_features(df_clean)

        if "Churn" in df_features.columns:
            df_features = df_features.drop(columns=["Churn"])

        preds = run_inference(model, df_features)
        preds["customerID"] = ids.values

        return {"predictions": preds.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
