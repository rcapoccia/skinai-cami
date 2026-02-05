# Stage 1: Builder con deps complete
FROM python:3.11-bookworm AS builder

RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime minimale
FROM python:3.11-slim-bookworm

# System deps TensorFlow/PIL (nomi Debian 12 corretti)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PORT=8080

EXPOSE 8080

# Use sh -c to expand $PORT variable at runtime
# Force rebuild: v2
CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --log-level info"
