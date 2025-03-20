#!/bin/bash
#
# Automated server setup script for crypto trading bot
# This script configures a fresh Ubuntu 22.04 server for running the bot
#

set -e  # Exit on any error

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting crypto trading bot server setup...${NC}"

# Make sure script is run as root
if [ "$(id -u)" -ne 0 ]; then
    echo -e "${RED}This script must be run as root${NC}" 
    exit 1
fi

# Get the username for the bot account
read -p "Enter username for bot account (default: botuser): " BOT_USER
BOT_USER=${BOT_USER:-botuser}

# Get the git repository URL
read -p "Enter git repository URL: " GIT_REPO
if [ -z "$GIT_REPO" ]; then
    echo -e "${RED}Git repository URL is required${NC}"
    exit 1
fi

# Update system
echo -e "${YELLOW}Updating system packages...${NC}"
apt update && apt upgrade -y

# Install dependencies
echo -e "${YELLOW}Installing required packages...${NC}"
apt install -y python3 python3-pip python3-venv git ufw fail2ban nginx certbot python3-certbot-nginx

# Configure firewall
echo -e "${YELLOW}Configuring firewall...${NC}"
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw --force enable

# Setup Fail2ban
echo -e "${YELLOW}Setting up Fail2ban...${NC}"
systemctl enable fail2ban
systemctl start fail2ban

# Create bot user
echo -e "${YELLOW}Creating bot user account...${NC}"
if id -u "$BOT_USER" >/dev/null 2>&1; then
    echo "User $BOT_USER already exists"
else
    useradd -m -s /bin/bash "$BOT_USER"
    echo -e "${YELLOW}Please set a password for $BOT_USER:${NC}"
    passwd "$BOT_USER"
fi

# Add to sudo group
usermod -aG sudo "$BOT_USER"

# Create application directory
echo -e "${YELLOW}Setting up application directory...${NC}"
APP_DIR="/home/$BOT_USER/crypto-bot"
if [ -d "$APP_DIR" ]; then
    echo "Directory $APP_DIR already exists"
else
    mkdir -p "$APP_DIR"
    chown "$BOT_USER:$BOT_USER" "$APP_DIR"
fi

# Clone repository
echo -e "${YELLOW}Cloning repository...${NC}"
cd /home/$BOT_USER
sudo -u "$BOT_USER" git clone "$GIT_REPO" "$APP_DIR"

# Setup Python virtual environment
echo -e "${YELLOW}Setting up Python environment...${NC}"
cd "$APP_DIR"
sudo -u "$BOT_USER" python3 -m venv venv
sudo -u "$BOT_USER" /bin/bash -c "source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"

# Create directories for data and logs
echo -e "${YELLOW}Creating data and log directories...${NC}"
sudo -u "$BOT_USER" mkdir -p "$APP_DIR/data"
sudo -u "$BOT_USER" mkdir -p "$APP_DIR/logs"
sudo -u "$BOT_USER" mkdir -p "$APP_DIR/reports"

# Create systemd service file
echo -e "${YELLOW}Creating systemd service...${NC}"
cat > /etc/systemd/system/crypto-bot.service << EOF
[Unit]
Description=Cryptocurrency Trading Bot
After=network.target

[Service]
User=$BOT_USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=append:$APP_DIR/logs/bot.log
StandardError=append:$APP_DIR/logs/error.log

[Install]
WantedBy=multi-user.target
EOF

# Setup cost monitoring
echo -e "${YELLOW}Setting up cost monitoring cron jobs...${NC}"
cat > /etc/cron.d/crypto-bot-monitoring << EOF
# Run cost monitoring daily at 1 AM
0 1 * * * $BOT_USER cd $APP_DIR && $APP_DIR/venv/bin/python scripts/cost_monitor.py >> $APP_DIR/logs/cost_monitor.log 2>&1

# Run fee tracker daily at 2 AM
0 2 * * * $BOT_USER cd $APP_DIR && $APP_DIR/venv/bin/python scripts/fee_tracker.py >> $APP_DIR/logs/fee_tracker.log 2>&1

# Run budget monitor on the 1st of each month
0 3 1 * * $BOT_USER cd $APP_DIR && $APP_DIR/venv/bin/python scripts/budget_monitor.py >> $APP_DIR/logs/budget_monitor.log 2>&1
EOF

chmod 644 /etc/cron.d/crypto-bot-monitoring

# Setup log rotation
echo -e "${YELLOW}Setting up log rotation...${NC}"
cat > /etc/logrotate.d/crypto-bot << EOF
$APP_DIR/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 $BOT_USER $BOT_USER
}
EOF

# Start the service
echo -e "${YELLOW}Starting crypto-bot service...${NC}"
systemctl daemon-reload
systemctl enable crypto-bot.service
systemctl start crypto-bot.service

# Configuration for environment variables (to be filled manually)
echo -e "${YELLOW}Creating environment variables template...${NC}"
cat > "$APP_DIR/.env.example" << EOF
# Exchange API Keys
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET=your_binance_secret
COINBASE_API_KEY=your_coinbase_api_key
COINBASE_SECRET=your_coinbase_secret

# Email Notification Settings
EMAIL_FROM=alerts@yourdomain.com
EMAIL_TO=you@yourdomain.com
EMAIL_PASSWORD=your_email_password
SMTP_SERVER=smtp.yourdomain.com
SMTP_PORT=587

# DigitalOcean API Keys (for billing)
DIGITALOCEAN_API_KEY=your_do_api_key
EOF

sudo -u "$BOT_USER" cp "$APP_DIR/.env.example" "$APP_DIR/.env"
echo -e "${YELLOW}Please edit $APP_DIR/.env to add your API keys and credentials${NC}"

# Setup Nginx as reverse proxy (optional)
read -p "Set up Nginx as reverse proxy? (y/n): " SETUP_NGINX
if [[ "$SETUP_NGINX" =~ ^[Yy]$ ]]; then
    read -p "Enter your domain name: " DOMAIN_NAME
    
    # Create Nginx config
    cat > /etc/nginx/sites-available/crypto-bot << EOF
server {
    listen 80;
    server_name $DOMAIN_NAME;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

    # Enable the site
    ln -sf /etc/nginx/sites-available/crypto-bot /etc/nginx/sites-enabled/
    
    # Test and reload Nginx
    nginx -t && systemctl reload nginx
    
    # Setup SSL with Let's Encrypt
    read -p "Set up SSL with Let's Encrypt? (y/n): " SETUP_SSL
    if [[ "$SETUP_SSL" =~ ^[Yy]$ ]]; then
        certbot --nginx -d $DOMAIN_NAME
    fi
fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "Bot service status: $(systemctl status crypto-bot.service | grep Active)"
echo -e "${YELLOW}Remember to edit $APP_DIR/.env to add your API keys and credentials${NC}"
echo -e "${YELLOW}Check logs at $APP_DIR/logs/bot.log${NC}" 