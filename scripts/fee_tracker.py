#!/usr/bin/env python3
"""
Trading fee tracker that estimates and logs monthly trading fees
from various exchanges.
"""

import os
import json
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='fee_tracker.log'
)
logger = logging.getLogger('fee_tracker')

# Data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Exchange API credentials (in production, store these securely)
EXCHANGE_CONFIGS = {
    'binance': {
        'apiKey': os.environ.get('BINANCE_API_KEY', ''),
        'secret': os.environ.get('BINANCE_SECRET', ''),
        'enableRateLimit': True
    },
    'coinbase': {
        'apiKey': os.environ.get('COINBASE_API_KEY', ''),
        'secret': os.environ.get('COINBASE_SECRET', ''),
        'enableRateLimit': True
    }
    # Add more exchanges as needed
}

def get_exchange_instance(exchange_id):
    """Create an instance of the specified exchange."""
    if exchange_id not in EXCHANGE_CONFIGS:
        logger.error(f"Exchange {exchange_id} is not configured")
        return None
    
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(EXCHANGE_CONFIGS[exchange_id])
        return exchange
    except Exception as e:
        logger.error(f"Failed to initialize exchange {exchange_id}: {str(e)}")
        return None

def fetch_trades(exchange, symbol, since=None, limit=1000):
    """Fetch trades from exchange for the given symbol."""
    try:
        if since is None:
            # Default to fetching trades for the last 30 days
            since = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
        
        trades = exchange.fetch_my_trades(symbol, since, limit)
        return trades
    except Exception as e:
        logger.error(f"Error fetching trades from {exchange.id} for {symbol}: {str(e)}")
        return []

def calculate_fees(trades):
    """Calculate total fees from a list of trades."""
    total_fees = 0
    fee_currencies = {}
    
    for trade in trades:
        # Some exchanges include fee in the trade info
        if 'fee' in trade and trade['fee'] is not None:
            fee = trade['fee']
            fee_cost = fee.get('cost', 0)
            fee_currency = fee.get('currency', 'USD')
            
            if fee_currency in fee_currencies:
                fee_currencies[fee_currency] += fee_cost
            else:
                fee_currencies[fee_currency] = fee_cost
            
            # Convert to USD if not already (simplified, in reality you'd use exchange rates)
            if fee_currency == 'USD':
                total_fees += fee_cost
    
    return {
        'total_fees_usd': total_fees,
        'fee_currencies': fee_currencies
    }

def save_fee_data(exchange_id, fee_data):
    """Save fee data to a JSON file for historical tracking."""
    fee_history_file = DATA_DIR / 'fee_history.json'
    fee_history = {}
    
    # Load existing data if available
    if fee_history_file.exists():
        with open(fee_history_file, 'r') as f:
            fee_history = json.load(f)
    
    # Add new data
    month_key = datetime.now().strftime('%Y-%m')
    
    if month_key not in fee_history:
        fee_history[month_key] = {}
    
    fee_history[month_key][exchange_id] = {
        'total_fees_usd': fee_data['total_fees_usd'],
        'fee_currencies': fee_data['fee_currencies'],
        'last_updated': datetime.now().isoformat()
    }
    
    # Save data
    with open(fee_history_file, 'w') as f:
        json.dump(fee_history, f, indent=2)
    
    logger.info(f"Fee data saved for {exchange_id} ({month_key})")

def get_all_trading_fees():
    """Fetch and calculate trading fees for all configured exchanges."""
    total_fees = 0
    fee_data_by_exchange = {}
    
    for exchange_id in EXCHANGE_CONFIGS:
        exchange = get_exchange_instance(exchange_id)
        if not exchange:
            continue
        
        # Get available trading pairs (simplified)
        try:
            markets = exchange.load_markets()
            symbols = list(markets.keys())
            
            # For demonstration, only use a few common pairs
            common_symbols = [s for s in symbols if 'BTC/USDT' in s or 'ETH/USDT' in s or 'ETH/BTC' in s]
            if not common_symbols and symbols:
                common_symbols = [symbols[0]]  # At least use one symbol
            
            all_trades = []
            for symbol in common_symbols:
                trades = fetch_trades(exchange, symbol)
                all_trades.extend(trades)
            
            fee_data = calculate_fees(all_trades)
            save_fee_data(exchange_id, fee_data)
            
            fee_data_by_exchange[exchange_id] = fee_data
            total_fees += fee_data['total_fees_usd']
            
            logger.info(f"Calculated fees for {exchange_id}: ${fee_data['total_fees_usd']:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing {exchange_id}: {str(e)}")
    
    return {
        'total_fees_usd': total_fees,
        'exchanges': fee_data_by_exchange
    }

def generate_fee_report():
    """Generate a monthly report of trading fees."""
    fee_history_file = DATA_DIR / 'fee_history.json'
    
    if not fee_history_file.exists():
        logger.warning("No fee history data available")
        return "No fee data available"
    
    with open(fee_history_file, 'r') as f:
        fee_history = json.load(f)
    
    # Create a DataFrame for analysis
    data = []
    
    for month, exchanges in fee_history.items():
        month_total = 0
        for exchange_id, fee_data in exchanges.items():
            month_total += fee_data['total_fees_usd']
            data.append({
                'month': month,
                'exchange': exchange_id,
                'fees_usd': fee_data['total_fees_usd']
            })
        
        # Add month total
        data.append({
            'month': month,
            'exchange': 'TOTAL',
            'fees_usd': month_total
        })
    
    if not data:
        return "No fee data available"
    
    df = pd.DataFrame(data)
    
    # Save report to CSV
    report_file = DATA_DIR / f'fee_report_{datetime.now().strftime("%Y%m%d")}.csv'
    df.to_csv(report_file, index=False)
    
    # Generate text report
    report = "Monthly Trading Fee Report\n"
    report += "=" * 30 + "\n\n"
    
    for month in sorted(fee_history.keys()):
        report += f"Month: {month}\n"
        report += "-" * 20 + "\n"
        
        month_total = 0
        for exchange_id, fee_data in fee_history[month].items():
            fee_amount = fee_data['total_fees_usd']
            month_total += fee_amount
            report += f"{exchange_id}: ${fee_amount:.2f}\n"
        
        report += f"Total: ${month_total:.2f}\n\n"
    
    return report

if __name__ == "__main__":
    logger.info("Starting trading fee tracking")
    
    # Fetch and calculate current trading fees
    fee_data = get_all_trading_fees()
    
    # Generate and display report
    report = generate_fee_report()
    print(report)
    
    logger.info(f"Total trading fees: ${fee_data['total_fees_usd']:.2f}")
    logger.info("Trading fee tracking completed") 