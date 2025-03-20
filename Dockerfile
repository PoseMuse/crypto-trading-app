FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app

# Create necessary directories
RUN mkdir -p data/sample output/example_backtest

# Default command (example: backtest demo)
CMD ["python", "src/backtest_example.py"] 