FROM python:3.14.0b1-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory inside container
WORKDIR /app

# Install system dependencies (for pandas, sklearn, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY forest.py .

# Create log/data dirs that your script expects
RUN mkdir -p /sniffer/logs /data

# By default run the RF script
CMD ["python", "forest.py"]
