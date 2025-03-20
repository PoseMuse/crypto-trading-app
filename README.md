# Crypto Trading Bot

A robust cryptocurrency trading platform with AI-based prediction models and DevOps infrastructure.

## Project Overview

This project implements a cryptocurrency trading bot with the following key features:

- Data collection from cryptocurrency exchanges via APIs
- Machine learning model pipeline for price predictions
- Sentiment analysis from social media feeds
- Backtesting framework for strategy validation
- Comprehensive DevOps infrastructure for deployment and monitoring

## Infrastructure & Deployment (Phase 7)

This repository includes a complete DevOps infrastructure for deploying and managing the crypto trading bot.

### Docker Setup

The project is containerized for easy deployment and consistency across environments:

```bash
# Build the Docker image
docker build -t crypto-bot:latest .

# Run the Docker container
docker run --rm crypto-bot:latest
```

### CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) automates testing and deployment:

- Runs tests on every push and pull request
- Builds Docker image on successful test completion
- Deploys to container registry when merging to main/master

### VPS Deployment

To deploy to a Virtual Private Server:

1. Set up a VPS with Docker installed (DigitalOcean, AWS Lightsail, etc.)
2. Clone the repository or upload the deployment files:
   ```bash
   git clone https://github.com/PoseMuse/crypto-trading-app.git /opt/crypto-bot
   ```
3. Configure environment variables in a `.env` file (copy from `.env.example`)
4. Run the deployment script:
   ```bash
   cd /opt/crypto-bot
   chmod +x scripts/deploy_vps.sh
   ./scripts/deploy_vps.sh
   ```

The deployment script:
- Installs Docker if not present
- Sets up project directories
- Builds and runs the Docker container
- Configures monitoring via cron jobs

### Monitoring

Monitoring is implemented through multiple mechanisms:

1. **System Monitoring Script**: Located at `scripts/monitor.sh`, runs every 5 minutes via cron to:
   - Check if the container is running
   - Restart if crashed (up to 3 attempts)
   - Send alerts if issues persist

2. **Health Check Endpoint**: An HTTP endpoint at port 8080:
   - Accessible via `http://YOUR_VPS_IP:8080/health`
   - Returns status information in JSON format
   - Can be monitored by external services like UptimeRobot

To set up monitoring manually:

```bash
# Configure cron job for monitoring
chmod +x /opt/crypto-bot/scripts/monitor.sh
crontab -e
# Add: */5 * * * * /opt/crypto-bot/scripts/monitor.sh
```

## Local Development

### Prerequisites

- Python 3.9+
- Docker (for containerized development)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/PoseMuse/crypto-trading-app.git
   cd crypto-trading-app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration (copy from `.env.example`)

### Running the Bot

```bash
# Run backtest example
python src/backtest_example.py

# Run in Docker
docker build -t crypto-bot:latest .
docker run --rm crypto-bot:latest
```

## Project Structure

```
.
├── .github/              # GitHub Actions workflows
├── config/               # Configuration files
├── data/                 # Data storage
├── docs/                 # Documentation
├── logs/                 # Log files
├── output/               # Output files (reports, graphs)
├── scripts/              # Deployment and utility scripts
├── src/                  # Source code
├── tests/                # Test suite
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore file
├── Dockerfile            # Docker configuration
├── README.md             # This file
├── docker-compose.yml    # Docker Compose configuration
└── requirements.txt      # Python dependencies
```

## Security Considerations

- API keys are stored in the `.env` file, which is not committed to git
- Separate environment files for development, testing, and production
- All scripts are executable only by the owner

## Future Enhancements

- Web dashboard for monitoring trading performance
- Secure key management using a secrets manager
- Advanced logging with Sentry or similar
- Enhanced metrics with Prometheus & Grafana
- Auto-scaling based on market volatility