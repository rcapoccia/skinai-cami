# SkinAI Backend

FastAPI server per analisi della pelle con modello ONNX.

## Setup Locale

```bash
# Crea virtual environment
python3 -m venv venv
source venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt

# Scarica il modello (da aggiungere)
mkdir -p models
# Copia skin-analyzer.onnx in models/

# Avvia server
python app.py
```

Server sarà disponibile su `http://localhost:8000`

## API Endpoints

### GET /health
Health check del server

```bash
curl http://localhost:8000/health
```

### POST /analyze
Analizza immagine e ritorna 5 score (0-10)

```bash
curl -X POST -F "file=@photo.jpg" http://localhost:8000/analyze
```

Response:
```json
{
  "status": "success",
  "scores": {
    "wrinkles": 6.3,
    "pores": 7.1,
    "spots": 3.8,
    "dark_circles": 5.0,
    "dehydration": 4.2
  },
  "metadata": {
    "model": "skin-analyzer-v1",
    "input_size": "224x224",
    "parameters": 5
  }
}
```

## Deploy su DigitalOcean

### 1. SSH nel droplet
```bash
ssh root@164.90.171.42
```

### 2. Installa Docker
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### 3. Clone repo
```bash
git clone https://github.com/rcapoccia/skinai-cami.git
cd skinai-cami
```

### 4. Copia il modello ONNX
```bash
mkdir -p models
# Copia skin-analyzer.onnx in models/
```

### 5. Build e Run Docker
```bash
docker build -t skinai-backend .
docker run -d -p 8000:8000 \
  -e ALLOWED_ORIGINS="https://tuodominio.vercel.app" \
  -v $(pwd)/models:/app/models \
  --name skinai-backend \
  skinai-backend
```

### 6. Verifica
```bash
curl http://localhost:8000/health
```

## Configurazione CORS

Modifica la variabile `ALLOWED_ORIGINS` per specificare i domini frontend:

```bash
docker run -d -p 8000:8000 \
  -e ALLOWED_ORIGINS="https://app.vercel.app,https://localhost:3000" \
  skinai-backend
```

## Variabili Ambiente

- `PORT`: Porta del server (default: 8000)
- `HOST`: Host del server (default: 0.0.0.0)
- `MODEL_PATH`: Path al modello ONNX (default: models/skin-analyzer.onnx)
- `ALLOWED_ORIGINS`: Domini CORS consentiti (default: *)

## Troubleshooting

### Modello non caricato
```
❌ Errore caricamento modello
```
Verifica che `models/skin-analyzer.onnx` esista e sia accessibile.

### Errore CORS
Se il frontend non riesce a connettersi, controlla:
1. La variabile `ALLOWED_ORIGINS`
2. L'URL del frontend nel browser
3. I log del container: `docker logs skinai-backend`

## Docs API

Accedi a `http://localhost:8000/docs` per la documentazione interattiva Swagger.
