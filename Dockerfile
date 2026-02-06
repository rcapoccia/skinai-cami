FROM tensorflow/tensorflow:2.16.1

WORKDIR /app

# Install additional dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY skinai_global_final.h5 .

# Expose port
EXPOSE 8080

# Run application
CMD ["python", "app.py"]
