FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK corpora inside container
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Copy full application source code
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "FastAPI.main:app", "--host", "0.0.0.0", "--port", "8000"]