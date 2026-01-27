FROM python:3.11-slim

WORKDIR /app

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Clone il repo da GitHub (sempre fresco)
RUN git clone https://github.com/rcapoccia/skinai-cami.git . && \
    git checkout HEAD

# Installa dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Esponi porta 9000
EXPOSE 9000

# Avvia app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
