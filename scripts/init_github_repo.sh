#!/bin/bash
# Initialize GitHub repository and make first commit

set -e

echo "Initializing GitHub repository for crypto trading bot..."

# Configure Git user
git config user.name "PoseMuse"
git config user.email "PoseMuseApp@gmail.com"

# Initialize repository if not already done
if [ ! -d ".git" ]; then
    git init
    echo "Git repository initialized."
else
    echo "Git repository already exists."
fi

# Check if remote already exists
if git remote | grep -q "origin"; then
    echo "Remote 'origin' already exists."
else
    echo "Adding remote 'origin'..."
    git remote add origin https://github.com/PoseMuse/crypto-trading-app.git
fi

# Add all files
git add .

# Commit changes
echo "Creating initial commit..."
git commit -m "Initial commit: Crypto Trading Bot infrastructure setup"

echo "Ready to push to GitHub."
echo "Run the following command to push:"
echo "git push -u origin main"
echo ""
echo "Note: You might need to authenticate with GitHub or use a personal access token." 