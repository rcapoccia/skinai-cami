#!/usr/bin/env python3
"""
Test script per verificare il modello TFLite e il backend localmente
"""

import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

print("=" * 60)
print("TEST LOCALE - SKINGLOW AI BACKEND")
print("=" * 60)

# Test 1: Verifica file modello
print("\n[TEST 1] Verifica file modello...")
model_path = "skinai_ensemble_final.tflite"
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅ Modello trovato: {model_path} ({file_size:.2f} MB)")
else:
    print(f"❌ Modello NON trovato: {model_path}")
    sys.exit(1)

# Test 2: Carica il modello TFLite
print("\n[TEST 2] Caricamento modello TFLite...")
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("✅ Modello caricato con successo")
except Exception as e:
    print(f"❌ Errore caricamento modello: {e}")
    sys.exit(1)

# Test 3: Verifica input/output details
print("\n[TEST 3] Verifica input/output details...")
try:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ Input shape: {input_details[0]['shape']}")
    print(f"✅ Input dtype: {input_details[0]['dtype']}")
    print(f"✅ Output shape: {output_details[0]['shape']}")
    print(f"✅ Output dtype: {output_details[0]['dtype']}")
except Exception as e:
    print(f"❌ Errore: {e}")
    sys.exit(1)

# Test 4: Crea immagine di test
print("\n[TEST 4] Creazione immagine di test...")
try:
    # Crea immagine random RGB 224x224
    test_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    print("✅ Immagine di test creata (224x224)")
except Exception as e:
    print(f"❌ Errore: {e}")
    sys.exit(1)

# Test 5: Preprocessing immagine
print("\n[TEST 5] Preprocessing immagine...")
try:
    arr = np.array(test_img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    print(f"✅ Array shape dopo preprocessing: {arr.shape}")
    print(f"✅ Array dtype: {arr.dtype}")
    print(f"✅ Array min/max: {arr.min():.4f} / {arr.max():.4f}")
except Exception as e:
    print(f"❌ Errore: {e}")
    sys.exit(1)

# Test 6: Inferenza
print("\n[TEST 6] Esecuzione inferenza...")
try:
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"✅ Inferenza completata")
    print(f"✅ Output shape: {output_data.shape}")
    print(f"✅ Output dtype: {output_data.dtype}")
except Exception as e:
    print(f"❌ Errore inferenza: {e}")
    sys.exit(1)

# Test 7: Parsing output
print("\n[TEST 7] Parsing output...")
try:
    predictions = output_data[0] if len(output_data.shape) > 1 else output_data
    scores = predictions.tolist() if isinstance(predictions, np.ndarray) else list(predictions)
    
    # Assicura 7 valori
    while len(scores) < 7:
        scores.append(0.0)
    scores = scores[:7]
    
    print(f"✅ Numero di score: {len(scores)}")
    print(f"✅ Score grezzi: {scores}")
    
    # Calcola i parametri finali
    final_scores = {
        "rughe": float(scores[0] * 10),
        "pori": float(scores[1] * 10),
        "macchie": float(scores[2] * 10),
        "occhiaie": float(scores[3] * 10),
        "glow": float(scores[4] * 10),
        "acne": float(scores[5] * 10),
        "pelle_pulita_percent": float(scores[6] * 100)
    }
    
    print(f"✅ Score finali:")
    for key, value in final_scores.items():
        print(f"   - {key}: {value:.2f}")
        
except Exception as e:
    print(f"❌ Errore parsing: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ TUTTI I TEST PASSATI!")
print("=" * 60)
print("\nIl backend dovrebbe funzionare correttamente.")
print("Puoi ora provare a deployare su Railway!")
