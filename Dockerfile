FROM python:3.11-slim

WORKDIR /app

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements
COPY requirements.txt .

# Installa dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia codice
COPY app.py .
COPY models/ models/

# Esponi porta 9000
EXPOSE 9000

# Avvia app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
