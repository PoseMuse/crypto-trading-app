# Crypto Trading Bot Web Dashboard

This module provides a web-based dashboard for monitoring the cryptocurrency trading bot, including portfolio performance, sentiment analysis, and trading controls.

## Features

- **Real-Time Monitoring**: View portfolio value, open positions, and recent trades
- **Sentiment Analysis Dashboard**: Visualize current market sentiment from multiple sources
- **Trading Controls**: Start and stop trading sessions with configurable parameters
- **Secure Access**: Basic Authentication to protect your dashboard
- **API Endpoints**: RESTful API for programmatic access to trading status

## Getting Started

### Local Development

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file (copy from `.env.example`) and configure:
   ```
   cp src/web_ui/.env.example src/web_ui/.env
   ```

3. Run the Flask application:
   ```
   python -m src.web_ui.app
   ```

4. Access the dashboard at http://localhost:5000

### Docker Deployment

1. Build and run with Docker Compose:
   ```
   docker-compose up -d
   ```

2. Access the dashboard at https://your-server-ip (if using Nginx with SSL)

## Security Recommendations

1. **Change Default Credentials**: Modify `DASHBOARD_USERNAME` and `DASHBOARD_PASSWORD` in the `.env` file
2. **Enable HTTPS**: Use the provided Nginx configuration for SSL/TLS
3. **Restrict Access**: Consider implementing IP restrictions in Nginx for additional security
4. **Regular Updates**: Keep dependencies updated to patch security vulnerabilities

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DASHBOARD_USERNAME` | Username for dashboard access | admin |
| `DASHBOARD_PASSWORD` | Password for dashboard access | *none* |
| `TRADING_PAIR` | Default trading pair to monitor | BTC/USDT |
| `INITIAL_CAPITAL` | Initial capital for paper trading | 10000.0 |
| `ENABLE_SENTIMENT` | Enable sentiment analysis | false |
| `PORT` | Web server port | 5000 |
| `FLASK_DEBUG` | Enable debug mode | false |

## API Endpoints

- **GET /api/status**: Get current trading status (portfolio value, positions, sentiment)
- **POST /start_trading**: Start a new trading session
- **POST /stop_trading**: Stop the current trading session

## Notes

- The dashboard automatically connects to the trading bot if it's running
- Sentiment data is refreshed every 30 minutes by default
- For production use, always deploy behind HTTPS using the provided Nginx configuration 