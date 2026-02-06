from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import gc
import os

# Import TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI(title="SkinGlow AI Backend")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load H5 model at startup
model = None

try:
    model_path = "skinai_global_final.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load Keras H5 model
    model = load_model(model_path)
    print(f"✅ H5 model loaded successfully: {model_path}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Error loading H5 model: {e}")
    model = None

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": "H5 (Keras)",
        "service": "SkinGlow AI Backend",
        "tensorflow_version": tf.__version__
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if model is None:
            return {
                "status": "error",
                "message": "Model not loaded"
            }
        
        # Read and process image
        img_data = await file.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img = img.resize((224, 224))
        
        # Normalize to match model input (0-1 float32)
        arr = np.array(img, dtype=np.float32) / 255.0
        # Add batch dimension
        arr = np.expand_dims(arr, 0)
        
        # Run inference
        predictions = model.predict(arr, verbose=0)
        
        # Get first batch predictions
        scores = predictions[0]
        
        # Cleanup
        gc.collect()
        
        return {
            "status": "success",
            "scores": {
                "rughe": float(scores[0]),
                "pori": float(scores[1]),
                "macchie": float(scores[2]),
                "occhiaie": float(scores[3]),
                "glow": float(scores[4]),
                "acne": float(scores[5]),
                "pelle_pulita_percent": float(scores[6]) if len(scores) > 6 else 0.0
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
