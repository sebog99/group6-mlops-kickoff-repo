# -------------------------------------------------------
# Base image: Miniconda with Python 3.10 (via environment.yml)
# -------------------------------------------------------
FROM continuumio/miniconda3:latest

WORKDIR /app

# -------------------------------------------------------
# Install dependencies from environment.yml
# -------------------------------------------------------
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

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
# Run the pipeline inside the conda environment
# -------------------------------------------------------
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mlops_project", "python", "-m", "src.main"]
