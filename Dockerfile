# Dockerfile

# Base image with Python
FROM python:3.11-slim

# Avoid writing .pyc files and enable stdout logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy everything else (models, code, API)
COPY . .

# Expose FastAPI on port 8000
EXPOSE 8000

# Default: run FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
