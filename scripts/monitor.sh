#!/bin/bash
# Simple monitoring script for crypto trading bot
# Recommended usage: Add to cron with: */5 * * * * /opt/crypto-bot/scripts/monitor.sh

# Configuration
CONTAINER_NAME="crypto-bot"
EMAIL_TO="alert@example.com"  # Replace this with your email in production
LOGS_DIR="/opt/crypto-bot/logs"
MAX_RESTART_ATTEMPTS=3
STATE_FILE="/tmp/crypto-bot-monitor"

# Initialize state file if it doesn't exist
if [ ! -f "$STATE_FILE" ]; then
    echo "0" > "$STATE_FILE"
fi

# Make sure logs directory exists
mkdir -p "$LOGS_DIR"

RESTART_COUNT=$(cat "$STATE_FILE")

# Create log entry with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOGS_DIR/monitor.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Send alert email
send_alert() {
    if command -v mail &> /dev/null; then
        echo "$1" | mail -s "ALERT: Crypto Bot Issue" "$EMAIL_TO"
        log "Alert email sent to $EMAIL_TO"
    else
        log "WARNING: mail command not available. Could not send alert. Install mailutils if needed."
        # Optional: Could add alternative notification methods here (e.g., curl to webhook)
    fi
}

# Health check via HTTP endpoint if available
check_health_endpoint() {
    if command -v curl &> /dev/null; then
        HEALTH_RESPONSE=$(curl -s http://localhost:8080/health)
        if [[ "$HEALTH_RESPONSE" == *"\"status\":\"ok\""* ]]; then
            log "Health check endpoint reports status OK"
            return 0
        else
            log "Health check endpoint reports issues or is unavailable"
            return 1
        fi
    else
        log "curl command not available. Skipping health endpoint check."
        return 0  # Not failing if curl isn't available
    fi
}

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    log "Container $CONTAINER_NAME is not running!"
    
    # Try to restart, but only up to MAX_RESTART_ATTEMPTS
    if [ "$RESTART_COUNT" -lt "$MAX_RESTART_ATTEMPTS" ]; then
        log "Attempting to restart container (attempt $((RESTART_COUNT+1))/$MAX_RESTART_ATTEMPTS)..."
        
        # Check if container exists but is stopped
        if docker ps -a | grep -q "$CONTAINER_NAME"; then
            docker start "$CONTAINER_NAME"
            log "Container $CONTAINER_NAME started."
        else
            log "Container $CONTAINER_NAME does not exist. Cannot restart."
            send_alert "Crypto bot container does not exist. Manual intervention required."
            exit 1
        fi
        
        # Increment restart count
        echo $((RESTART_COUNT+1)) > "$STATE_FILE"
    else
        log "Maximum restart attempts ($MAX_RESTART_ATTEMPTS) reached."
        send_alert "Crypto bot container failed to restart after $MAX_RESTART_ATTEMPTS attempts. Manual intervention required."
        exit 1
    fi
else
    # Container is running, reset restart count
    echo "0" > "$STATE_FILE"
    log "Container $CONTAINER_NAME is running."
    
    # Check container health via logs
    CONTAINER_LOGS=$(docker logs --tail 20 "$CONTAINER_NAME" 2>&1)
    if echo "$CONTAINER_LOGS" | grep -q "ERROR"; then
        log "Errors detected in container logs."
        # Uncomment to enable error alerts
        send_alert "Errors detected in crypto bot container logs: $(echo "$CONTAINER_LOGS" | grep "ERROR" | head -n 3)"
    fi
    
    # Check health endpoint
    check_health_endpoint
fi

exit 0 