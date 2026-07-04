FROM python:3.11-slim

# Install system dependencies (ffmpeg is required for video decoding)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CPU/CUDA support 
# (Installing the PyTorch index first ensures correct binary resolution)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the workspace (models, data, etc.)
# We do this after requirements so Docker caches the pip install layer
COPY . /app

# Ensure outputs directories exist
RUN mkdir -p /app/outputs/uploads /app/outputs/api_results

# Expose FastAPI port
EXPOSE 8000

# Start Uvicorn
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
