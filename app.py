from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
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

# Load TFLite model at startup
interpreter = None
input_details = None
output_details = None

try:
    # Load TFLite model
    model_path = "skinai_ensemble_final.tflite"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ TFLite model loaded successfully: {model_path}")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")
    interpreter = None

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": interpreter is not None,
        "model_type": "TFLite",
        "service": "SkinGlow AI Backend"
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if interpreter is None:
            return {
                "status": "error",
                "message": "Model not loaded"
            }
        
        # Read and process image
        img_data = await file.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img = img.resize((224, 224))
        
        # Normalize to match model input
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)
        
        # Run TFLite inference
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions = output_data[0] if len(output_data.shape) > 1 else output_data
        
        # Extract scores (assuming 7 outputs)
        scores = predictions.tolist() if isinstance(predictions, np.ndarray) else list(predictions)
        
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
