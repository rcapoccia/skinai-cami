FROM python:3.11

# Install system dependencies for computer vision and TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    tensorflow==2.16.1 \
    pillow \
    numpy \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    google-generativeai

WORKDIR /app
COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
