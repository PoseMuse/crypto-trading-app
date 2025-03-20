#!/bin/bash
# Setup script to initialize directories for the crypto trading bot
# Run this once after cloning the repository

set -e

echo "Setting up crypto trading bot directories..."

# Create necessary directories
mkdir -p data/sample
mkdir -p output/example_backtest
mkdir -p logs
mkdir -p config

# Copy example env file to config directory
if [ -f .env.example ]; then
    cp .env.example config/.env.example
    echo "Copied .env.example to config directory"
    echo "Remember to create your own config/.env file with your API keys"
fi

# Create empty log file
touch logs/setup.log
echo "$(date): Setup script ran successfully" >> logs/setup.log

echo "Setup completed successfully. Directory structure created."
echo ""
echo "Next steps:"
echo "1. Create a config/.env file with your API keys (use config/.env.example as a template)"
echo "2. Run 'python src/health_check.py' to verify everything is working correctly"
echo "3. Test the Docker build with 'docker build -t crypto-bot:latest .'" 