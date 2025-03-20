#!/bin/bash
# Deployment script for crypto trading bot to a VPS
# This script should be run on the VPS

set -e

# Configuration
CONTAINER_NAME="crypto-bot"
IMAGE_NAME="ghcr.io/YOUR_USERNAME/crypto-bot:latest"
DATA_DIR="/opt/crypto-bot/data"
CONFIG_DIR="/opt/crypto-bot/config"
LOG_DIR="/opt/crypto-bot/logs"

echo "Starting deployment of $CONTAINER_NAME..."

# Make sure we have directories for persistent data
mkdir -p $DATA_DIR $CONFIG_DIR $LOG_DIR

# Pull the latest image
echo "Pulling latest Docker image..."
docker pull $IMAGE_NAME

# Stop and remove existing container if it exists
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "Stopping and removing existing container..."
    docker stop $CONTAINER_NAME || true
    docker rm $CONTAINER_NAME || true
fi

# Create and start the new container
echo "Creating and starting new container..."
docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -v $DATA_DIR:/app/data \
    -v $CONFIG_DIR:/app/config \
    -v $LOG_DIR:/app/logs \
    --env-file $CONFIG_DIR/.env \
    $IMAGE_NAME

echo "Deployment completed successfully!"
echo "You can check the logs with: docker logs $CONTAINER_NAME" 