# VPS Deployment Guide

This guide walks through deploying the crypto trading bot and web dashboard on a Virtual Private Server (VPS).

## Prerequisites

- A VPS with at least 1GB RAM (2GB recommended)
- Ubuntu 20.04 LTS or newer
- Docker and Docker Compose installed
- A domain name (optional, but recommended for HTTPS)

## Step 1: Set Up the VPS

### Install Docker and Docker Compose

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker repository
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Install Docker
sudo apt update
sudo apt install -y docker-ce

# Add your user to the docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.15.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installations
docker --version
docker-compose --version
```

## Step 2: Clone the Repository

```bash
# Navigate to your preferred directory
cd /opt

# Clone the repository
sudo git clone https://github.com/your-username/crypto-trading-bot.git
cd crypto-trading-bot

# Set appropriate permissions
sudo chown -R $USER:$USER /opt/crypto-trading-bot
```

## Step 3: Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit the .env file with your configuration
nano .env
```

Set appropriate values for:
- API keys for exchanges (if using real trading)
- Dashboard credentials
- Trading parameters
- Email notifications (if needed)

## Step 4: Set Up SSL Certificates

### Option 1: Use Let's Encrypt (recommended for production)

If you have a domain pointing to your VPS:

```bash
# Install certbot
sudo apt install -y certbot

# Get certificates (replace with your domain)
sudo certbot certonly --standalone -d yourdomain.com

# Copy certificates to Nginx directory
mkdir -p nginx/ssl
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem nginx/ssl/crypto-bot.crt
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem nginx/ssl/crypto-bot.key
sudo chown -R $USER:$USER nginx/ssl
```

### Option 2: Use Self-Signed Certificates (for testing only)

```bash
# Generate self-signed certificates
cd nginx
./generate_self_signed_cert.sh
cd ..
```

## Step 5: Update Nginx Configuration

Edit `nginx/conf.d/crypto-bot.conf` to set your server name:

```nginx
server_name yourdomain.com;
```

## Step 6: Deploy with Docker Compose

```bash
# Build and start the services
docker-compose up -d

# Check logs to make sure everything is working
docker-compose logs -f
```

## Step 7: Secure the Server

```bash
# Install and configure UFW firewall
sudo apt install -y ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https
sudo ufw enable

# Check firewall status
sudo ufw status
```

## Access the Dashboard

- If using a domain: https://yourdomain.com
- If using IP only: https://your-server-ip

Use the username and password defined in the `.env` file.

## Monitoring and Maintenance

### View Docker Logs

```bash
# View all logs
docker-compose logs

# View logs for a specific service
docker-compose logs crypto-bot-ui

# Follow logs in real-time
docker-compose logs -f
```

### Update the Application

```bash
# Pull latest changes
git pull

# Rebuild and restart containers
docker-compose down
docker-compose up -d --build
```

### Backup Data

```bash
# Backup data directory
tar -czvf crypto-bot-backup-$(date +%Y%m%d).tar.gz data output logs
```

## Troubleshooting

### Service Not Starting

Check logs for errors:
```bash
docker-compose logs crypto-bot-ui
```

### HTTPS Not Working

Verify certificates are correctly mounted:
```bash
docker-compose exec nginx ls -la /etc/nginx/ssl
```

### Dashboard Not Accessible

Check if the services are running:
```bash
docker-compose ps
```

## Cost-Effective Hosting Options

For a minimal setup that stays within a $5-10/month budget:

- DigitalOcean: Basic Droplet ($5/month, 1GB RAM)
- Linode: Nanode ($5/month, 1GB RAM)
- Vultr: Cloud Compute ($3.50/month, 1GB RAM)
- Hetzner: CX11 (€3.49/month, 2GB RAM)

These options should provide sufficient resources for running the bot with the web dashboard. 