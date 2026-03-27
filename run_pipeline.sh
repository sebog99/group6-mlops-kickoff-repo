#!/bin/bash
# -------------------------------------------------------
# run_pipeline.sh
# Cleans, rebuilds, and runs the Telco Churn pipeline
#
# Usage:
#   bash run_pipeline.sh          # run the training pipeline (default)
#   bash run_pipeline.sh api      # run the FastAPI server on port 8000
# -------------------------------------------------------

IMAGE_NAME="telco-churn"
CONTAINER_NAME="telco-churn-container"
MODE=${1:-pipeline}

echo ">>> Removing old container..."
docker rm -f $CONTAINER_NAME 2>/dev/null || true

echo ">>> Removing old image..."
docker rmi -f $IMAGE_NAME:latest 2>/dev/null || true

echo ">>> Building image..."
docker build -t $IMAGE_NAME:latest .

if [ "$MODE" = "api" ]; then
    echo ">>> Running API server on port 8000..."
    docker run \
      --name $CONTAINER_NAME \
      --env-file .env \
      -p 8000:8000 \
      -v "$(pwd)/models:/app/models" \
      -v "$(pwd)/logs:/app/logs" \
      $IMAGE_NAME:latest \
      conda run --no-capture-output -n mlops_project uvicorn src.api:app --host 0.0.0.0 --port 8000
else
    echo ">>> Running training pipeline..."
    docker run \
      --name $CONTAINER_NAME \
      --env-file .env \
      -v "$(pwd)/data:/app/data" \
      -v "$(pwd)/models:/app/models" \
      -v "$(pwd)/reports:/app/reports" \
      -v "$(pwd)/logs:/app/logs" \
      $IMAGE_NAME:latest
fi

echo ">>> Done."
