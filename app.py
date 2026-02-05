from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import gc
import os

app = FastAPI(title="SkinGlow AI Backend")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
try:
    # Try to load the global model (ensemble)
    model_path = "skinai_global_final.h5"
    if not os.path.exists(model_path):
        # Fallback to any available H5 model
        h5_files = [f for f in os.listdir(".") if f.endswith(".h5")]
        if h5_files:
            model_path = h5_files[0]
            print(f"Using model: {model_path}")
        else:
            raise FileNotFoundError("No H5 model found in directory")
    
    model = load_model(model_path)
    print(f"✅ Model loaded successfully: {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "service": "SkinGlow AI Backend"
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Read and process image
        img_data = await file.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img = img.resize((224, 224))
        
        # Normalize
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)
        
        # Predict
        predictions = model.predict(arr, verbose=0)
        
        # Extract scores (assuming 7 outputs)
        scores = predictions[0].tolist() if len(predictions.shape) > 1 else predictions.tolist()
        
        # Ensure we have 7 values
        while len(scores) < 7:
            scores.append(0.0)
        scores = scores[:7]
        
        # Cleanup
        gc.collect()
        
        return {
            "status": "success",
            "scores": {
                "rughe": float(scores[0] * 10),
                "pori": float(scores[1] * 10),
                "macchie": float(scores[2] * 10),
                "occhiaie": float(scores[3] * 10),
                "glow": float(scores[4] * 10),
                "acne": float(scores[5] * 10),
                "pelle_pulita_percent": float(scores[6] * 100)
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
