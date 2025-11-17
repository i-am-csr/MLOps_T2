FROM python:3.9-slim

# Avoid .pyc files and use unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set the working directory
WORKDIR /app

# Install system dependencies (for pandas, numpy, xgboost, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------
# 2) Copy application code + model artifacts (.pkl)
# --------------------------------------------------------------------
# This copies:
# - main.py
# - proyecto_final/ package
# - models/ directory with all .pkl files
# - any other project files needed at runtime
COPY . /app

# Make sure the models directory exists (in case of CI quirks)
RUN mkdir -p /app/models

# --------------------------------------------------------------------
# 3) Expose port and run FastAPI with Uvicorn
# --------------------------------------------------------------------
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
