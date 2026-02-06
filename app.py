from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import gc
import os
import onnxruntime as ort

app = FastAPI(title="SkinGlow AI Backend")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX model at startup
session = None

try:
    model_path = "skin_analyzer.onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    session = ort.InferenceSession(model_path)
    print(f"✅ ONNX model loaded successfully: {model_path}")
    print(f"   Input name: {session.get_inputs()[0].name}")
    print(f"   Output names: {[o.name for o in session.get_outputs()]}")
except Exception as e:
    print(f"❌ Error loading ONNX model: {e}")
    session = None

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": session is not None,
        "model_type": "ONNX",
        "service": "SkinGlow AI Backend"
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if session is None:
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
        # Transpose to CHW format (ONNX expects C, H, W)
        arr = np.transpose(arr, (2, 0, 1))
        # Add batch dimension
        arr = np.expand_dims(arr, 0)
        
        # Run ONNX inference
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: arr})
        
        # Get predictions (6 values: wrinkles, pores, spots, dark_circles, dehydration, acne)
        predictions = output[0][0]  # Shape: (6,)
        
        # Cleanup
        gc.collect()
        
        return {
            "status": "success",
            "scores": {
                "rughe": float(predictions[0]),
                "pori": float(predictions[1]),
                "macchie": float(predictions[2]),
                "occhiaie": float(predictions[3]),
                "disidratazione": float(predictions[4]),
                "acne": float(predictions[5])
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
