#!/usr/bin/env python3
"""
Health check script for crypto trading bot.
This script can be used to verify the system is functioning properly.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def check_data_directories():
    """Check if data directories exist and are accessible."""
    directories = [
        'data',
        'data/sample',
        'output',
        'output/example_backtest',
        'logs'
    ]
    
    issues = []
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                issues.append(f"Failed to create directory {directory}: {str(e)}")
        elif not os.access(dir_path, os.R_OK | os.W_OK):
            issues.append(f"Directory {directory} exists but is not readable/writable")
    
    return {
        "status": "ok" if not issues else "warning",
        "issues": issues
    }

def check_python_modules():
    """Check if all required Python modules are installed correctly."""
    required_modules = [
        'numpy', 'pandas', 'matplotlib', 'scipy', 'scikit-learn',
        'python-dotenv', 'requests', 'ccxt', 'nltk', 'ta'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    return {
        "status": "ok" if not missing_modules else "error",
        "missing_modules": missing_modules
    }

def check_last_run():
    """Check when the system was last run successfully."""
    log_files = list(Path('logs').glob('*.log')) if Path('logs').exists() else []
    
    if not log_files:
        return {
            "status": "warning",
            "message": "No log files found, system may have never been run"
        }
    
    # Get the most recent log file
    latest_log = max(log_files, key=os.path.getmtime, default=None)
    
    if latest_log:
        last_modified = datetime.fromtimestamp(os.path.getmtime(latest_log))
        time_diff = datetime.now() - last_modified
        
        # If log was modified in the last 24 hours, system is probably ok
        if time_diff.total_seconds() < 86400:  # 24 hours in seconds
            return {
                "status": "ok",
                "last_run": last_modified.isoformat(),
                "log_file": str(latest_log)
            }
        else:
            return {
                "status": "warning",
                "message": f"Last log update was {time_diff.days} days ago",
                "last_run": last_modified.isoformat(),
                "log_file": str(latest_log)
            }
    
    return {
        "status": "error",
        "message": "Unable to determine last run time"
    }

def check_environment_variables():
    """Check if required environment variables are set."""
    required_vars = [
        'BINANCE_API_KEY',
        'BINANCE_SECRET'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    return {
        "status": "ok" if not missing_vars else "warning",
        "missing_vars": missing_vars
    }

def run_health_check():
    """Run all health checks and return results."""
    start_time = time.time()
    
    health_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",  # Replace with actual version
        "checks": {
            "data_directories": check_data_directories(),
            "python_modules": check_python_modules(),
            "last_run": check_last_run(),
            "environment": check_environment_variables()
        }
    }
    
    # Calculate overall status
    statuses = [check["status"] for check in health_data["checks"].values()]
    if "error" in statuses:
        health_data["status"] = "error"
    elif "warning" in statuses:
        health_data["status"] = "warning"
    else:
        health_data["status"] = "ok"
    
    health_data["duration_ms"] = int((time.time() - start_time) * 1000)
    
    return health_data

def main():
    """Main function to run health check and output results."""
    logger.info("Running health check...")
    health_data = run_health_check()
    
    # Pretty-print the results
    print(json.dumps(health_data, indent=2))
    
    # Return exit code based on status
    if health_data["status"] == "error":
        return 2  # Error
    elif health_data["status"] == "warning":
        return 1  # Warning
    else:
        return 0  # OK

if __name__ == "__main__":
    sys.exit(main()) 