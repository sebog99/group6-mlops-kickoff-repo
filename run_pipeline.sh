#!/bin/bash
# -------------------------------------------------------
# run_pipeline.sh
# Cleans, rebuilds, and runs the Telco Churn pipeline
# -------------------------------------------------------

IMAGE_NAME="telco-churn"
CONTAINER_NAME="telco-churn-container"

echo ">>> Removing old container..."
docker rm -f $CONTAINER_NAME 2>/dev/null || true

echo ">>> Removing old image..."
docker rmi -f $IMAGE_NAME:latest 2>/dev/null || true

echo ">>> Building image..."
docker build -t $IMAGE_NAME:latest .

echo ">>> Running pipeline..."
docker run \
  --name $CONTAINER_NAME \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/reports:/app/reports" \
  -v "$(pwd)/logs:/app/logs" \
  $IMAGE_NAME:latest

echo ">>> Done."
