#!/usr/bin/env python3
"""
Simple HTTP server that provides health check endpoints for the crypto trading bot.
Use this to monitor the bot with external services like UptimeRobot.

Run with: python health_check_endpoint.py
"""

import os
import json
import socket
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 8080  # Port to serve on
START_TIME = datetime.now()

class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health check endpoints."""
    
    def _set_headers(self, status_code=200, content_type='application/json'):
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()
    
    def _build_response(self):
        """Build health check response."""
        uptime = (datetime.now() - START_TIME).total_seconds()
        
        # Get system metrics
        memory_info = {}
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = {
                "memory_percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent(interval=0.1)
            }
        except ImportError:
            memory_info = {"error": "psutil module not available"}
        
        return {
            "status": "ok",
            "version": "1.0.0",
            "uptime_seconds": uptime,
            "hostname": socket.gethostname(),
            "timestamp": datetime.now().isoformat(),
            "system": memory_info
        }
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health' or self.path == '/':
            self._set_headers()
            response = self._build_response()
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/ping':
            self._set_headers(content_type='text/plain')
            self.wfile.write(b'pong')
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

def run_server():
    """Run the HTTP server."""
    server = HTTPServer((HOST, PORT), HealthCheckHandler)
    logger.info(f"Starting health check server on http://{HOST}:{PORT}")
    logger.info("Available endpoints: / or /health, /ping")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        server.server_close()

if __name__ == "__main__":
    # Start the server in a thread so it can be integrated with the main bot
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    logger.info("Health check server running in background")
    
    # The following is only for standalone operation
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting")
        sys.exit(0) 