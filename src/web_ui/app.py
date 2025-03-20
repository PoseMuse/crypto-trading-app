"""
Flask web application for the Crypto Trading Bot dashboard.

This module provides a web interface to monitor trading activities,
portfolio performance, and sentiment analysis data.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_httpauth import HTTPBasicAuth
import os
import sys
import logging
from datetime import datetime
import json
from pathlib import Path

# Add the parent directory to the path to import from sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('crypto_bot_ui')

# Try to import LiveTrader, but handle import errors gracefully
try:
    from src.paper_trading.live_trader import LiveTrader
    LIVE_TRADER_AVAILABLE = True
    logger.info("LiveTrader module successfully imported")
except ImportError as e:
    LIVE_TRADER_AVAILABLE = False
    logger.warning(f"LiveTrader module not available: {e}")
    logger.warning("Running in demo mode with mocked data")
    
    # Create a mock LiveTrader class for demo purposes
    class LiveTrader:
        def __init__(self, symbol="BTC/USDT", cash=10000.0, enable_sentiment=False):
            self.symbol = symbol
            self.cash = cash
            self.enable_sentiment = enable_sentiment
            self.cerebro = type('obj', (object,), {
                'broker': type('obj', (object,), {
                    'getvalue': lambda: cash
                })
            })
            
        def get_real_time_sentiment(self):
            return {
                'compound_score': 0.25,
                'sentiment': 'positive',
                'source_breakdown': {
                    'reddit': {'compound_score': 0.3, 'count': 15},
                    'twitter': {'compound_score': 0.2, 'count': 25},
                    'telegram': {'compound_score': 0.1, 'count': 5}
                }
            }
        
        def get_current_status(self):
            return {
                'portfolio_value': self.cash,
                'positions': [],
                'sentiment': self.get_real_time_sentiment(),
                'recent_trades': [],
                'symbol': self.symbol,
                'enable_sentiment': self.enable_sentiment,
                'timestamp': datetime.now().isoformat()
            }

# Initialize Flask app
app = Flask(__name__)
auth = HTTPBasicAuth()

# Security configuration
USERS = {
    os.environ.get('DASHBOARD_USERNAME', 'admin'): 
    os.environ.get('DASHBOARD_PASSWORD', 'password123')
}

# Trader instance (initialize when first needed)
trader = None
trader_lock = False

@auth.verify_password
def verify_password(username, password):
    """Verify username and password for HTTP Basic Auth."""
    if username in USERS and USERS[username] == password:
        return username
    return None

def get_trader():
    """Get or initialize the trader instance."""
    global trader, trader_lock
    
    if trader is None and not trader_lock:
        try:
            trader_lock = True  # Prevent multiple initializations
            logger.info("Initializing LiveTrader for UI...")
            trader = LiveTrader(
                symbol=os.environ.get('TRADING_PAIR', 'BTC/USDT'),
                cash=float(os.environ.get('INITIAL_CAPITAL', '10000.0')),
                enable_sentiment=os.environ.get('ENABLE_SENTIMENT', 'false').lower() == 'true'
            )
            trader_lock = False
            logger.info("LiveTrader initialized successfully")
        except Exception as e:
            trader_lock = False
            logger.error(f"Error initializing trader: {e}")
            return None
    
    return trader

@app.route('/')
@auth.login_required
def index():
    """Render the main dashboard page."""
    try:
        # Try to get trader instance, but don't block if not available
        t = get_trader()
        
        if t and hasattr(t, 'cerebro') and hasattr(t.cerebro, 'broker'):
            portfolio_value = t.cerebro.broker.getvalue()
        else:
            portfolio_value = float(os.environ.get('INITIAL_CAPITAL', '10000.0'))
        
        # Get sentiment data if available
        if t and t.enable_sentiment:
            sentiment_info = t.get_real_time_sentiment()
        else:
            sentiment_info = {
                "compound_score": 0.0,
                "sentiment": "neutral",
                "source_breakdown": {}
            }
        
        # Get positions if available
        positions = []
        if t and hasattr(t, 'cerebro') and hasattr(t.cerebro, 'broker'):
            for data in t.cerebro.datas:
                position = t.cerebro.broker.getposition(data)
                if position.size != 0:
                    positions.append({
                        'symbol': data._name,
                        'size': position.size,
                        'price': position.price,
                        'value': position.size * position.price
                    })
        
        return render_template(
            'index.html',
            portfolio_value=portfolio_value,
            sentiment=sentiment_info,
            positions=positions,
            trader_status="Active" if t else "Demo Mode" if not LIVE_TRADER_AVAILABLE else "Inactive",
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            demo_mode=not LIVE_TRADER_AVAILABLE
        )
        
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return render_template(
            'index.html',
            error=str(e),
            portfolio_value=10000.0,
            sentiment={"compound_score": 0.0, "sentiment": "neutral", "source_breakdown": {}},
            positions=[],
            trader_status="Error",
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            demo_mode=not LIVE_TRADER_AVAILABLE
        )

@app.route('/api/status')
@auth.login_required
def api_status():
    """API endpoint to get the current trading status."""
    try:
        t = get_trader()
        
        if t is None:
            return jsonify({
                "status": "inactive",
                "error": "Trader not initialized",
                "demo_mode": not LIVE_TRADER_AVAILABLE
            }), 503
        
        # Get current status data
        status_data = {
            "status": "active",
            "portfolio_value": t.cerebro.broker.getvalue(),
            "sentiment": t.get_real_time_sentiment() if t.enable_sentiment else {"compound_score": 0.0},
            "positions": [],
            "timestamp": datetime.now().isoformat(),
            "demo_mode": not LIVE_TRADER_AVAILABLE
        }
        
        # Add position data
        if hasattr(t.cerebro, 'datas'):
            for data in t.cerebro.datas:
                position = t.cerebro.broker.getposition(data)
                if position.size != 0:
                    status_data["positions"].append({
                        "symbol": data._name,
                        "size": position.size,
                        "price": position.price,
                        "value": position.size * position.price
                    })
        
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f"Error in API status: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "demo_mode": not LIVE_TRADER_AVAILABLE
        }), 500

@app.route('/start_trading', methods=['POST'])
@auth.login_required
def start_trading():
    """Start paper trading with specified parameters."""
    try:
        global trader
        
        if not LIVE_TRADER_AVAILABLE:
            return jsonify({"error": "LiveTrader module not available", "demo_mode": True}), 400
        
        if trader is not None:
            return jsonify({"error": "Trading already in progress"}), 400
        
        # Get parameters from form
        symbol = request.form.get('symbol', os.environ.get('TRADING_PAIR', 'BTC/USDT'))
        cash = float(request.form.get('cash', os.environ.get('INITIAL_CAPITAL', '10000.0')))
        enable_sentiment = request.form.get('enable_sentiment', 'false').lower() == 'true'
        
        # Initialize trader
        trader = LiveTrader(
            symbol=symbol,
            cash=cash,
            enable_sentiment=enable_sentiment
        )
        
        # Start paper trading in a separate thread
        # In a real app, you'd use a proper background task system
        # For simplicity in this example, we're just initializing the trader
        
        return jsonify({"status": "success", "message": "Trading started"})
        
    except Exception as e:
        logger.error(f"Error starting trading: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stop_trading', methods=['POST'])
@auth.login_required
def stop_trading():
    """Stop the current trading session."""
    global trader
    
    if not LIVE_TRADER_AVAILABLE:
        return jsonify({"error": "LiveTrader module not available", "demo_mode": True}), 400
    
    if trader is None:
        return jsonify({"error": "No trading in progress"}), 400
    
    # Clean up and stop the trader
    trader = None
    
    return jsonify({"status": "success", "message": "Trading stopped"})

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {e}")
    return render_template('500.html', error=str(e)), 500

if __name__ == "__main__":
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=port, debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true') 