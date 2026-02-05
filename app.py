from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import gc
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SkinGlow AI Backend")

interpreter = None

try:
    logger.info("Loading TFLite model...")
    interpreter = tf.lite.Interpreter("skinai_ensemble_final.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("✅ TFLite model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading TFLite: {e}")
    interpreter = None

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": interpreter is not None}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not interpreter:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})
    
    try:
        content = await file.read()
        logger.info(f"Processing file: {file.filename}, size: {len(content)} bytes")
        
        img = Image.open(io.BytesIO(content)).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)
        
        logger.info(f"Running inference...")
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        scores = output[0].tolist()
        logger.info(f"Inference complete: {scores}")
        
        # Ensure we have 7 scores
        while len(scores) < 7:
            scores.append(0.0)
        scores = scores[:7]
        
        gc.collect()
        
        return {
            "rughe": float(scores[0]),
            "pori": float(scores[1]),
            "macchie": float(scores[2]),
            "occhiaie": float(scores[3]),
            "glow": float(scores[4]),
            "acne": float(scores[5]),
            "pelle_pulita_percent": float(scores[6])
        }
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        gc.collect()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
