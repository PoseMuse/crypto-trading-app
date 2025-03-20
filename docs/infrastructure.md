# Infrastructure & DevOps Guide

This document describes the infrastructure and DevOps setup for the crypto trading bot.

## Overview

The crypto trading bot is containerized using Docker and can be deployed to any Linux-based virtual private server (VPS). The setup includes:

1. Docker-based deployment
2. GitHub Actions for CI/CD
3. Health checks and monitoring
4. VPS deployment

## Local Development

### Prerequisites

- Docker
- Python 3.9+
- Git

### Running Locally

```bash
# Clone the repository
git clone https://github.com/your-username/crypto-bot.git
cd crypto-bot

# Build and run using Docker
docker build -t crypto-bot:latest .
docker run --rm crypto-bot:latest
```

### Running with Docker Compose

```bash
# Start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

## CI/CD with GitHub Actions

The repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that:

1. Runs tests on every push and pull request
2. Builds and pushes Docker images when code is merged to main/master branch

### Environment Variables & Secrets

To enable integration tests with real API keys, configure the following secrets in your GitHub repository:

- `BINANCE_API_KEY`
- `BINANCE_SECRET`

## VPS Deployment

### Recommended VPS Specs

- DigitalOcean or AWS Lightsail ($5-$10/month)
- Ubuntu 20.04 LTS
- 1GB RAM minimum
- 25GB SSD storage

### Deployment Process

1. Set up a VPS with SSH access
2. Copy the deployment script to the VPS:
   ```bash
   scp scripts/deploy_vps.sh user@your-vps-ip:~/
   ```
3. Run the deployment script:
   ```bash
   ssh user@your-vps-ip
   chmod +x deploy_vps.sh
   ./deploy_vps.sh
   ```

The script will:
- Install Docker if needed
- Clone the repository
- Build the Docker image
- Start the container with proper volumes
- Set up a monitoring cron job

## Monitoring & Health Checks

### Health Check Endpoint

The bot includes a health check HTTP server (`src/health_check_endpoint.py`) that provides:

- `/health` - Returns detailed status information
- `/ping` - Simple ping endpoint

To access the health check when the bot is running:
```bash
curl http://your-vps-ip:8080/health
```

### Monitoring Script

A monitoring script (`scripts/monitor.sh`) is included that:

1. Checks if the container is running
2. Restarts it if necessary
3. Sends email alerts when problems occur
4. Logs all actions

The script is set up as a cron job to run every 5 minutes.

### External Monitoring

For more robust monitoring, consider using:

- UptimeRobot - Free service to monitor the HTTP endpoint
- Prometheus + Grafana - Advanced metrics and dashboards
- Sentry - Error tracking

## Security Considerations

1. **API Keys**: Store API keys securely using environment variables
2. **VPS Access**: Use SSH keys and disable password authentication
3. **Updates**: Regularly update the VPS and Docker images
4. **Firewall**: Configure UFW to only allow necessary ports (SSH, health check)

## Common Issues

### Docker Container Won't Start

Check the Docker logs:
```bash
docker logs crypto-bot
```

### API Connection Issues

Verify your API keys are correctly set in the `.env` file.

### Memory Issues

If the VPS runs out of memory, consider upgrading or adding swap space:
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
``` 