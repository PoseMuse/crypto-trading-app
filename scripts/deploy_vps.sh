#!/bin/bash
# Deployment script for crypto trading bot on VPS
# Usage: ./deploy_vps.sh

set -e

echo "Starting deployment of crypto trading bot..."

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker $USER
    echo "Docker installed successfully"
else
    echo "Docker is already installed"
fi

# Create project directory
PROJECT_DIR="/opt/crypto-bot"
sudo mkdir -p $PROJECT_DIR
sudo chown $USER:$USER $PROJECT_DIR

# Create data directories
mkdir -p $PROJECT_DIR/data
mkdir -p $PROJECT_DIR/logs
mkdir -p $PROJECT_DIR/config
mkdir -p $PROJECT_DIR/output

# Clone repository or pull latest changes
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "Updating existing repository..."
    cd $PROJECT_DIR
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/PoseMuse/crypto-trading-app.git $PROJECT_DIR
    cd $PROJECT_DIR
    git config user.name "PoseMuse"
    git config user.email "PoseMuseApp@gmail.com"
fi

# Copy .env file if it exists
if [ -f ".env" ]; then
    cp .env $PROJECT_DIR/.env
    echo "Copied .env file to project directory"
else
    echo "WARNING: .env file not found. Please create one in $PROJECT_DIR/"
fi

# Build Docker image
echo "Building Docker image..."
cd $PROJECT_DIR
docker build -t crypto-bot:latest .

# Stop and remove existing container if it exists
if docker ps -a | grep -q crypto-bot; then
    echo "Stopping and removing existing container..."
    docker stop crypto-bot || true
    docker rm crypto-bot || true
fi

# Run Docker container
echo "Starting container..."
docker run -d \
    --name crypto-bot \
    --restart unless-stopped \
    -p 8080:8080 \
    -v $PROJECT_DIR/data:/app/data \
    -v $PROJECT_DIR/logs:/app/logs \
    -v $PROJECT_DIR/.env:/app/.env \
    -v $PROJECT_DIR/output:/app/output \
    crypto-bot:latest

# Set up cron job for monitoring
echo "Setting up monitoring cron job..."
MONITOR_SCRIPT="$PROJECT_DIR/scripts/monitor.sh"
chmod +x $MONITOR_SCRIPT

# Add cron job if not already present
CRON_JOB="*/5 * * * * $MONITOR_SCRIPT > $PROJECT_DIR/logs/monitor_cron.log 2>&1"
(crontab -l 2>/dev/null | grep -v "$MONITOR_SCRIPT" || true; echo "$CRON_JOB") | crontab -

echo "Deployment completed successfully!"
echo "Container logs: docker logs crypto-bot"
echo "Health check endpoint: http://YOUR_VPS_IP:8080/health" 