# Crypto Trading Bot Deployment Guide

This guide outlines the steps to deploy the crypto trading bot infrastructure (Phase 7). Follow these instructions to set up both local development environments and production deployments.

## Prerequisites

Before starting, ensure you have:

- Git installed
- Docker and Docker Compose installed
- Python 3.9+ installed
- A GitHub account with access to the repository
- SSH keys configured for your GitHub account (for deployment)
- A VPS with Ubuntu 20.04/22.04 (for production deployment)

## Local Setup and Verification

### 1. Clone the Repository

```bash
git clone https://github.com/PoseMuse/crypto-trading-app.git
cd crypto-trading-app
```

### 2. Set Up Environment

Create a `.env` file based on the example:

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Build and Test Docker Container

```bash
# Build the Docker image
docker build -t crypto-bot:latest .

# Run the container in test mode
docker run --rm crypto-bot:latest
```

The container should run the sample backtest script (`src/backtest_example.py`) and exit successfully.

### 4. Verify the Docker Compose Setup

```bash
# Start all services
docker-compose up -d

# Check container logs
docker-compose logs -f

# Stop all services
docker-compose down
```

## GitHub Repository Setup

### 1. Initialize Repository

If setting up from scratch:

```bash
# Run the initialization script
./scripts/init_github_repo.sh

# Push to GitHub
git push -u origin main
```

### 2. Configure GitHub Secrets

In the GitHub repository settings, add these secrets for CI/CD:

- `BINANCE_API_KEY`: Your Binance API key
- `BINANCE_SECRET`: Your Binance API secret
- `GITHUB_TOKEN`: Automatically provided by GitHub

## VPS Deployment (Production)

### 1. Provision a VPS

Recommended specifications:
- 1 vCPU
- 2GB RAM
- 25GB SSD storage
- Ubuntu 22.04 LTS

### 2. Initial Server Setup

SSH into your server and perform initial setup:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install basic tools
sudo apt install -y git curl wget htop

# Configure firewall (optional but recommended)
sudo ufw allow OpenSSH
sudo ufw allow 8080/tcp  # For health check endpoint
sudo ufw enable
```

### 3. Install Docker on VPS

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to the docker group
sudo usermod -aG docker $USER

# Apply changes (you may need to log out and back in)
newgrp docker
```

### 4. Deploy the Bot

Option 1: Using the deployment script:

```bash
# Clone the repository
git clone https://github.com/PoseMuse/crypto-trading-app.git /opt/crypto-bot
cd /opt/crypto-bot

# Make scripts executable
chmod +x scripts/*.sh

# Set up configuration
cp .env.example .env
nano .env  # Edit with your actual configuration

# Run deployment script
./scripts/deploy_vps.sh
```

Option 2: Manual deployment:

```bash
# Clone the repository
git clone https://github.com/PoseMuse/crypto-trading-app.git /opt/crypto-bot
cd /opt/crypto-bot

# Set up configuration
cp .env.example .env
nano .env  # Edit with your actual configuration

# Build and run with Docker Compose
docker-compose up -d
```

### 5. Set Up Monitoring

The deployment script already sets up basic monitoring via cron jobs. Verify it's working:

```bash
# Check cron jobs
crontab -l

# Verify monitor script is executable
chmod +x /opt/crypto-bot/scripts/monitor.sh

# Run monitor script manually to test
/opt/crypto-bot/scripts/monitor.sh
```

### 6. Verify Deployment

```bash
# Check if the container is running
docker ps

# Check container logs
docker logs crypto-bot

# Test health check endpoint
curl http://localhost:8080/health
```

## Updating the Bot

### 1. Update Local Repository

```bash
cd /opt/crypto-bot
git pull
```

### 2. Rebuild and Restart

```bash
# Option 1: Using deploy script
./scripts/deploy_vps.sh

# Option 2: Manual update
docker-compose down
docker-compose build
docker-compose up -d
```

## Troubleshooting

### Container Not Starting

Check Docker logs:

```bash
docker logs crypto-bot
```

### Health Check Not Working

Verify the health check endpoint is running:

```bash
# Check if port 8080 is being listened on
sudo netstat -tulpn | grep 8080

# Verify the container is exposing the port
docker ps
```

### Monitoring Issues

Check monitor logs:

```bash
cat /opt/crypto-bot/logs/monitor.log
```

## Backup and Restore

### Backup Data

```bash
# Create a backup directory
mkdir -p /backup/crypto-bot

# Backup data and configuration
cp -r /opt/crypto-bot/data /backup/crypto-bot/
cp -r /opt/crypto-bot/logs /backup/crypto-bot/
cp /opt/crypto-bot/.env /backup/crypto-bot/
```

### Restore from Backup

```bash
# Restore from backup
cp -r /backup/crypto-bot/data /opt/crypto-bot/
cp -r /backup/crypto-bot/logs /opt/crypto-bot/
cp /backup/crypto-bot/.env /opt/crypto-bot/
```

## Security Considerations

- Never commit `.env` files with API keys or secrets
- Use SSH keys instead of passwords for server access
- Keep the server updated with security patches
- Consider using a secrets manager for production API keys
- Isolate the trading bot from other services on your VPS

## Advanced Configuration

### Using a Custom Domain

If you want to use a domain for your health check endpoint:

1. Set up DNS records pointing to your VPS
2. Install and configure Nginx:

```bash
sudo apt install -y nginx
sudo nano /etc/nginx/sites-available/crypto-bot

# Add configuration:
# server {
#     listen 80;
#     server_name your-domain.com;
#
#     location / {
#         proxy_pass http://localhost:8080;
#         proxy_set_header Host $host;
#         proxy_set_header X-Real-IP $remote_addr;
#     }
# }

sudo ln -s /etc/nginx/sites-available/crypto-bot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

3. Set up SSL with Let's Encrypt:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Setting Up Email Alerts

To enable email alerts, install the `mailutils` package:

```bash
sudo apt install -y mailutils
```

Configure it to use your SMTP server or a service like SendGrid. 