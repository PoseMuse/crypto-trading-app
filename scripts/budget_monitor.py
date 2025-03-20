#!/usr/bin/env python3
"""
Budget monitoring script that aggregates all costs (hosting, trading fees, etc.)
and sends alerts when approaching the annual budget threshold.
"""

import os
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='budget_monitor.log'
)
logger = logging.getLogger('budget_monitor')

# Data and output directories
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Email configuration
EMAIL_FROM = os.environ.get('EMAIL_FROM', 'alerts@yourdomain.com')
EMAIL_TO = os.environ.get('EMAIL_TO', 'you@yourdomain.com')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'your_email_password')
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.yourdomain.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))

# Budget thresholds (in USD)
ANNUAL_BUDGET = 2000.00
WARNING_THRESHOLD = 0.8  # 80% of budget

def load_cost_data():
    """Load cost data from various sources."""
    # Initialize the cost structure
    costs = {
        'hosting': {},
        'trading_fees': {},
        'other': {}
    }
    
    # Load hosting costs
    hosting_history_file = DATA_DIR / 'cost_history.json'
    if hosting_history_file.exists():
        with open(hosting_history_file, 'r') as f:
            costs['hosting'] = json.load(f)
    
    # Load trading fees
    fee_history_file = DATA_DIR / 'fee_history.json'
    if fee_history_file.exists():
        with open(fee_history_file, 'r') as f:
            costs['trading_fees'] = json.load(f)
    
    # Load other costs (if any)
    other_costs_file = DATA_DIR / 'other_costs.json'
    if other_costs_file.exists():
        with open(other_costs_file, 'r') as f:
            costs['other'] = json.load(f)
    
    return costs

def calculate_monthly_costs(costs, year, month):
    """Calculate total costs for a specific month."""
    month_key = f"{year}-{month:02d}"
    
    # Initialize totals
    total = {
        'hosting': 0.0,
        'trading_fees': 0.0,
        'other': 0.0,
        'total': 0.0
    }
    
    # Sum hosting costs
    if month_key in costs['hosting']:
        total['hosting'] = costs['hosting'][month_key].get('hosting_cost', 0.0)
    
    # Sum trading fees
    if month_key in costs['trading_fees']:
        for exchange, fee_data in costs['trading_fees'][month_key].items():
            if exchange != 'TOTAL':  # Avoid double counting if we've stored totals
                total['trading_fees'] += fee_data.get('total_fees_usd', 0.0)
    
    # Sum other costs
    if month_key in costs['other']:
        for cost_type, amount in costs['other'][month_key].items():
            total['other'] += amount
    
    # Calculate total
    total['total'] = total['hosting'] + total['trading_fees'] + total['other']
    
    return total

def calculate_ytd_costs():
    """Calculate year-to-date costs."""
    costs = load_cost_data()
    
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    monthly_costs = []
    ytd_total = 0.0
    
    for month in range(1, current_month + 1):
        month_costs = calculate_monthly_costs(costs, current_year, month)
        monthly_costs.append({
            'year': current_year,
            'month': month,
            'hosting': month_costs['hosting'],
            'trading_fees': month_costs['trading_fees'],
            'other': month_costs['other'],
            'total': month_costs['total']
        })
        ytd_total += month_costs['total']
    
    return {
        'monthly_costs': monthly_costs,
        'ytd_total': ytd_total
    }

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

def generate_budget_report(ytd_data):
    """Generate a comprehensive budget report with visualizations."""
    # Create a DataFrame for analysis
    df = pd.DataFrame(ytd_data['monthly_costs'])
    
    # Create a summary report
    report_date = datetime.now().strftime('%Y-%m-%d')
    summary = f"Budget Report - {report_date}\n"
    summary += "=" * 50 + "\n\n"
    
    summary += f"Annual Budget: ${ANNUAL_BUDGET:.2f}\n"
    summary += f"Year-to-Date Spending: ${ytd_data['ytd_total']:.2f}\n"
    summary += f"Remaining Budget: ${ANNUAL_BUDGET - ytd_data['ytd_total']:.2f}\n"
    summary += f"Budget Utilization: {(ytd_data['ytd_total'] / ANNUAL_BUDGET) * 100:.1f}%\n\n"
    
    summary += "Monthly Breakdown:\n"
    summary += "-" * 50 + "\n"
    summary += "Month\tHosting\t\tTrading Fees\tOther\t\tTotal\n"
    
    for _, row in df.iterrows():
        month_name = datetime(row['year'], row['month'], 1).strftime('%b')
        summary += f"{month_name}\t${row['hosting']:.2f}\t${row['trading_fees']:.2f}\t\t${row['other']:.2f}\t${row['total']:.2f}\n"
    
    # Create visualizations
    try:
        # Monthly costs stacked bar chart
        plt.figure(figsize=(10, 6))
        df['month_name'] = df.apply(lambda row: datetime(row['year'], row['month'], 1).strftime('%b'), axis=1)
        
        # Create stacked bar
        plt.bar(df['month_name'], df['hosting'], label='Hosting')
        plt.bar(df['month_name'], df['trading_fees'], bottom=df['hosting'], label='Trading Fees')
        plt.bar(df['month_name'], df['other'], bottom=df['hosting'] + df['trading_fees'], label='Other')
        
        plt.xlabel('Month')
        plt.ylabel('Cost (USD)')
        plt.title('Monthly Costs Breakdown')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        chart_file = REPORTS_DIR / f'monthly_costs_{report_date}.png'
        plt.savefig(chart_file)
        
        # Create pie chart for overall distribution
        plt.figure(figsize=(8, 8))
        labels = ['Hosting', 'Trading Fees', 'Other']
        sizes = [df['hosting'].sum(), df['trading_fees'].sum(), df['other'].sum()]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Cost Distribution')
        
        # Save pie chart
        pie_chart_file = REPORTS_DIR / f'cost_distribution_{report_date}.png'
        plt.savefig(pie_chart_file)
        
        summary += f"\nVisualizations saved to {REPORTS_DIR}\n"
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        summary += "\nError creating visualizations\n"
    
    # Save report to file
    report_file = REPORTS_DIR / f'budget_report_{report_date}.txt'
    with open(report_file, 'w') as f:
        f.write(summary)
    
    return summary

def check_budget_threshold():
    """Check if current spending is approaching budget threshold."""
    ytd_data = calculate_ytd_costs()
    ytd_total = ytd_data['ytd_total']
    
    # Calculate what percentage of the annual budget has been used
    budget_utilization = ytd_total / ANNUAL_BUDGET
    
    # Check if we're approaching or exceeding threshold
    if budget_utilization >= WARNING_THRESHOLD:
        subject = f"WARNING: Annual budget at {budget_utilization * 100:.1f}% utilization"
        message = f"""
        Your annual budget utilization has reached {budget_utilization * 100:.1f}%.
        
        Annual Budget: ${ANNUAL_BUDGET:.2f}
        YTD Spending: ${ytd_total:.2f}
        Remaining Budget: ${ANNUAL_BUDGET - ytd_total:.2f}
        
        Please review your spending to avoid exceeding the annual budget.
        """
        
        send_email_alert(subject, message)
        logger.warning(f"Budget threshold warning: {budget_utilization * 100:.1f}% utilized")
        
        if budget_utilization >= 1.0:
            # We've exceeded the budget
            over_budget_subject = "ALERT: Annual budget exceeded"
            over_budget_message = f"""
            Your annual budget of ${ANNUAL_BUDGET:.2f} has been exceeded.
            Current spending: ${ytd_total:.2f}
            Over budget by: ${ytd_total - ANNUAL_BUDGET:.2f}
            
            Please take immediate action to address this issue.
            """
            
            send_email_alert(over_budget_subject, over_budget_message)
            logger.error(f"Budget exceeded: ${ytd_total:.2f} spent of ${ANNUAL_BUDGET:.2f} budget")
    
    return {
        'budget': ANNUAL_BUDGET,
        'ytd_total': ytd_total,
        'utilization': budget_utilization
    }

def record_other_cost(amount, description):
    """Record miscellaneous costs not captured by other tracking."""
    other_costs_file = DATA_DIR / 'other_costs.json'
    other_costs = {}
    
    # Load existing data if available
    if other_costs_file.exists():
        with open(other_costs_file, 'r') as f:
            other_costs = json.load(f)
    
    # Add new data
    month_key = datetime.now().strftime('%Y-%m')
    
    if month_key not in other_costs:
        other_costs[month_key] = {}
    
    timestamp = datetime.now().isoformat()
    cost_id = f"cost_{timestamp}"
    
    other_costs[month_key][cost_id] = {
        'amount': amount,
        'description': description,
        'date': timestamp
    }
    
    # Save data
    with open(other_costs_file, 'w') as f:
        json.dump(other_costs, f, indent=2)
    
    logger.info(f"Recorded other cost: ${amount:.2f} - {description}")

if __name__ == "__main__":
    logger.info("Starting budget monitoring")
    
    # Check budget threshold
    budget_status = check_budget_threshold()
    
    # Generate detailed report
    ytd_data = calculate_ytd_costs()
    report = generate_budget_report(ytd_data)
    
    print(report)
    
    logger.info(f"Budget check completed. Utilization: {budget_status['utilization'] * 100:.1f}%")
    
    # Uncomment to add a sample other cost for testing
    # record_other_cost(25.00, "SSL Certificate Renewal") 