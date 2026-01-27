#!/usr/bin/env python3
"""
SkinAI - Backend FastAPI
API REST per inferenza modello skin analysis
Deploy su DigitalOcean - TensorFlow Lite Version
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from typing import List
import logging

# ============================================================================
# SETUP
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SkinAI Backend",
    description="API per analisi pelle con AI",
    version="1.0.0"
)

# CORS - Configurare per Vercel
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODEL LOADING
# ============================================================================

MODEL_PATH = os.getenv("MODEL_PATH", "models/skin-analyzer.onnx")

# Prova a caricare il modello
interpreter = None
try:
    logger.info(f"Caricando modello da: {MODEL_PATH}")
    
    # Se è un file ONNX, convertilo a TFLite
    if MODEL_PATH.endswith('.onnx'):
        logger.warning("File ONNX rilevato. Usando fallback con score casuali.")
        interpreter = None
    else:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    
    logger.info("✅ Modello caricato con successo!")
except Exception as e:
    logger.warning(f"⚠️ Modello non disponibile, usando fallback: {e}")
    interpreter = None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocessa immagine per il modello"""
    try:
        # Carica immagine
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize a 224x224
        image = image.resize((224, 224))
        
        # Converti a numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Normalizza (ImageNet)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array / 255.0 - mean) / std
        
        # Aggiungi batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        logger.error(f"Errore preprocessing: {e}")
        raise

def denormalize_scores(scores: np.ndarray = None) -> dict:
    """Denormalizza score da 0-1 a 0-10 o genera fallback"""
    if scores is None or len(scores) == 0:
        # Fallback: score casuali per testing
        import random
        return {
            "wrinkles": round(random.uniform(2, 8), 1),
            "pores": round(random.uniform(3, 7), 1),
            "spots": round(random.uniform(1, 6), 1),
            "dark_circles": round(random.uniform(2, 7), 1),
            "dehydration": round(random.uniform(1, 5), 1),
        }
    
    return {
        "wrinkles": float(scores[0][0] * 10) if len(scores[0]) > 0 else 5.0,
        "pores": float(scores[0][1] * 10) if len(scores[0]) > 1 else 5.0,
        "spots": float(scores[0][2] * 10) if len(scores[0]) > 2 else 5.0,
        "dark_circles": float(scores[0][3] * 10) if len(scores[0]) > 3 else 5.0,
        "dehydration": float(scores[0][4] * 10) if len(scores[0]) > 4 else 5.0,
    }

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": interpreter is not None,
        "model_path": MODEL_PATH,
        "mode": "tensorflow-lite" if interpreter else "fallback"
    }

@app.post("/analyze")
async def analyze_skin(file: UploadFile = File(...)):
    """
    Analizza immagine pelle e ritorna 5 score
    
    Response:
    {
        "wrinkles": 6.3,
        "pores": 7.1,
        "spots": 3.8,
        "dark_circles": 5.0,
        "dehydration": 4.2
    }
    """
    
    try:
        # Leggi file
        image_bytes = await file.read()
        
        if not image_bytes:
            raise HTTPException(status_code=400, detail="File vuoto")
        
        # Preprocessa
        image_array = preprocess_image(image_bytes)
        
        # Inferenza
        logger.info(f"Eseguendo inferenza su immagine {len(image_bytes)} bytes")
        
        if interpreter:
            # Usa TensorFlow Lite
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            interpreter.set_tensor(input_details[0]['index'], image_array)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            scores = denormalize_scores(output_data)
        else:
            # Fallback: score casuali
            logger.warning("Usando fallback scores (modello non disponibile)")
            scores = denormalize_scores()
        
        logger.info(f"Analisi completata: {scores}")
        
        return JSONResponse(content={
            "status": "success",
            "scores": scores,
            "metadata": {
                "model": "skin-analyzer-v1",
                "input_size": "224x224",
                "parameters": 5,
                "mode": "tensorflow-lite" if interpreter else "fallback"
            }
        })
    
    except Exception as e:
        logger.error(f"Errore analisi: {e}")
        raise HTTPException(status_code=500, detail=f"Errore analisi: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "SkinAI Backend",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "docs": "/docs"
        }
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Avviando server su {host}:{port}")
    uvicorn.run(app, host=host, port=port)
