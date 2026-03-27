# -------------------------------------------------------
# Base image: Miniconda with Python 3.10 (via environment.yml)
# -------------------------------------------------------
FROM continuumio/miniconda3:latest

WORKDIR /app

# -------------------------------------------------------
# Install dependencies from conda-lock.yml (pinned, reproducible)
# -------------------------------------------------------
COPY environment.yml .
COPY conda-lock.yml .
RUN conda install -n base -c conda-forge conda-lock -y && \
    conda-lock install -n mlops_project conda-lock.yml && \
    conda clean -afy

# -------------------------------------------------------
# Copy source code and configuration
# -------------------------------------------------------
COPY src/ src/
COPY tests/ tests/
COPY config.yaml .
COPY conftest.py .

# -------------------------------------------------------
# Create runtime directories (data and models are mounted as volumes)
# -------------------------------------------------------
RUN mkdir -p data/raw data/processed data/inference models reports logs

# -------------------------------------------------------
# Expose API port
# -------------------------------------------------------
EXPOSE 8000

# -------------------------------------------------------
# Default: run the training pipeline
# Override with CMD to run the API instead:
#   docker run ... telco-churn:latest conda run -n mlops_project uvicorn src.api:app --host 0.0.0.0 --port 8000
# -------------------------------------------------------
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mlops_project", "python", "-m", "src.main"]
