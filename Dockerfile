FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for spaCy and LanguageTool
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    openjdk-21-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Copy application code
COPY python_api.py .

# Expose port
ENV PORT=8080
EXPOSE 8080

# Run the application
CMD exec uvicorn python_api:app --host 0.0.0.0 --port ${PORT}
