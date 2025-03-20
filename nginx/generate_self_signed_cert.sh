#!/bin/bash
# Script to generate self-signed SSL certificates for development

# Create SSL directory if it doesn't exist
mkdir -p ssl

# Generate a private key
openssl genrsa -out ssl/crypto-bot.key 2048

# Generate a Certificate Signing Request (CSR)
openssl req -new -key ssl/crypto-bot.key -out ssl/crypto-bot.csr -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Generate self-signed certificate (valid for 365 days)
openssl x509 -req -days 365 -in ssl/crypto-bot.csr -signkey ssl/crypto-bot.key -out ssl/crypto-bot.crt

# Remove the CSR file as it's no longer needed
rm ssl/crypto-bot.csr

echo "Self-signed certificate generated successfully!"
echo "Files created:"
echo "  - ssl/crypto-bot.key (private key)"
echo "  - ssl/crypto-bot.crt (certificate)"
echo
echo "Note: This certificate is self-signed and will generate browser warnings."
echo "For production use, replace with a certificate from a trusted CA." 