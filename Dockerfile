# Stage 1: Build
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

# Copy only necessary files from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set environment variables
ENV PATH=/root/.local/bin:$PATH

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]