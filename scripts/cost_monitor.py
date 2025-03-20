#!/usr/bin/env python3
"""
Cost monitoring script that tracks DigitalOcean hosting costs and sends alerts 
when approaching budget thresholds.
"""

import os
import json
import requests
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='cost_monitor.log'
)
logger = logging.getLogger('cost_monitor')

# Configuration (in production, store these securely)
DIGITALOCEAN_API_KEY = os.environ.get('DIGITALOCEAN_API_KEY', 'your_do_api_key')
EMAIL_FROM = os.environ.get('EMAIL_FROM', 'alerts@yourdomain.com')
EMAIL_TO = os.environ.get('EMAIL_TO', 'you@yourdomain.com')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'your_email_password')
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.yourdomain.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))

# Budget thresholds (in USD)
ANNUAL_BUDGET = 2000.00
MONTHLY_WARNING_THRESHOLD = ANNUAL_BUDGET / 12 * 0.8  # 80% of monthly budget

def get_digitalocean_billing():
    """Fetch billing information from DigitalOcean API."""
    headers = {
        'Authorization': f'Bearer {DIGITALOCEAN_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    # Get current billing cycle
    now = datetime.now()
    first_day_month = datetime(now.year, now.month, 1)
    
    try:
        # DigitalOcean billing API endpoint
        response = requests.get(
            'https://api.digitalocean.com/v2/customers/my/billing_history',
            headers=headers
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Calculate current month's spending
        monthly_billing = 0
        for item in data.get('billing_history', []):
            item_date = datetime.fromisoformat(item['date'].replace('Z', '+00:00'))
            if item_date >= first_day_month:
                monthly_billing += item['amount']
        
        return {
            'current_month_cost': monthly_billing,
            'month': now.strftime('%B %Y')
        }
    
    except requests.RequestException as e:
        logger.error(f"Error fetching DigitalOcean billing: {str(e)}")
        return None

def send_email_alert(subject, message):
    """Send email alert when cost thresholds are reached."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Alert email sent: {subject}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to send email alert: {str(e)}")
        return False

def check_monthly_costs():
    """Check monthly costs and alert if approaching threshold."""
    billing_info = get_digitalocean_billing()
    if not billing_info:
        logger.error("Could not retrieve billing information")
        return
    
    monthly_cost = billing_info['current_month_cost']
    month = billing_info['month']
    
    logger.info(f"Current monthly cost for {month}: ${monthly_cost:.2f}")
    
    # Check if we're approaching monthly threshold
    if monthly_cost >= MONTHLY_WARNING_THRESHOLD:
        percentage = (monthly_cost / (ANNUAL_BUDGET / 12)) * 100
        subject = f"WARNING: Monthly hosting costs at {percentage:.1f}% of budget"
        message = f"""
        Your DigitalOcean hosting costs for {month} have reached ${monthly_cost:.2f},
        which is {percentage:.1f}% of your monthly budget allocation.
        
        Monthly budget: ${ANNUAL_BUDGET/12:.2f}
        Current spending: ${monthly_cost:.2f}
        
        Please review your resource usage to avoid exceeding the annual budget of ${ANNUAL_BUDGET:.2f}.
        """
        
        send_email_alert(subject, message)
        
        # Also log to console for cron job visibility
        print(subject)

def save_cost_data(billing_info):
    """Save cost data to a JSON file for historical tracking."""
    if not billing_info:
        return
    
    cost_history_file = 'cost_history.json'
    cost_data = {}
    
    # Load existing data if available
    if os.path.exists(cost_history_file):
        with open(cost_history_file, 'r') as f:
            cost_data = json.load(f)
    
    # Add new data
    month_key = datetime.now().strftime('%Y-%m')
    cost_data[month_key] = {
        'hosting_cost': billing_info['current_month_cost'],
        'last_updated': datetime.now().isoformat()
    }
    
    # Save data
    with open(cost_history_file, 'w') as f:
        json.dump(cost_data, f, indent=2)
    
    logger.info(f"Cost data saved for {month_key}")

if __name__ == "__main__":
    logger.info("Starting cost monitoring check")
    billing_info = get_digitalocean_billing()
    
    if billing_info:
        check_monthly_costs()
        save_cost_data(billing_info)
    
    logger.info("Cost monitoring check completed") 