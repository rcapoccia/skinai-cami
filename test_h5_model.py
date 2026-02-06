#!/usr/bin/env python3
"""
Test script per verificare il modello H5
"""

import os
import sys
import numpy as np
from PIL import Image

print("=" * 60)
print("TEST H5 MODEL - SKINGLOW AI")
print("=" * 60)

# Test 1: Verifica file modello
print("\n[TEST 1] Verifica file modello...")
model_path = "skinai_global_final.h5"
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅ Modello trovato: {model_path} ({file_size:.2f} MB)")
else:
    print(f"❌ Modello NON trovato: {model_path}")
    sys.exit(1)

# Test 2: Carica TensorFlow
print("\n[TEST 2] Caricamento TensorFlow...")
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    print(f"✅ TensorFlow versione: {tf.__version__}")
except ImportError as e:
    print(f"❌ Errore import TensorFlow: {e}")
    sys.exit(1)

# Test 3: Carica il modello H5
print("\n[TEST 3] Caricamento modello H5...")
try:
    model = load_model(model_path)
    print("✅ Modello caricato con successo")
except Exception as e:
    print(f"❌ Errore caricamento modello: {e}")
    sys.exit(1)

# Test 4: Verifica input/output details
print("\n[TEST 4] Verifica input/output details...")
try:
    print(f"✅ Input shape: {model.input_shape}")
    print(f"✅ Output shape: {model.output_shape}")
    print(f"✅ Numero parametri: {model.count_params()}")
except Exception as e:
    print(f"❌ Errore: {e}")
    sys.exit(1)

# Test 5: Crea immagine di test
print("\n[TEST 5] Creazione immagine di test...")
try:
    # Crea immagine random RGB 224x224
    test_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    print("✅ Immagine di test creata (224x224)")
except Exception as e:
    print(f"❌ Errore: {e}")
    sys.exit(1)

# Test 6: Preprocessing immagine
print("\n[TEST 6] Preprocessing immagine...")
try:
    arr = np.array(test_img, dtype=np.float32) / 255.0
    # Add batch dimension
    arr = np.expand_dims(arr, 0)
    print(f"✅ Array shape dopo preprocessing: {arr.shape}")
    print(f"✅ Array dtype: {arr.dtype}")
    print(f"✅ Array min/max: {arr.min():.4f} / {arr.max():.4f}")
except Exception as e:
    print(f"❌ Errore: {e}")
    sys.exit(1)

# Test 7: Inferenza
print("\n[TEST 7] Esecuzione inferenza...")
try:
    predictions = model.predict(arr, verbose=0)
    print(f"✅ Inferenza completata")
    print(f"✅ Output shape: {predictions.shape}")
    print(f"✅ Output dtype: {predictions.dtype}")
except Exception as e:
    print(f"❌ Errore inferenza: {e}")
    sys.exit(1)

# Test 8: Parsing output
print("\n[TEST 8] Parsing output...")
try:
    scores = predictions[0]  # Get first batch
    
    print(f"✅ Numero di parametri: {len(scores)}")
    print(f"✅ Valori grezzi: {scores}")
    
    # Calcola i parametri finali
    final_scores = {
        "rughe": float(scores[0]),
        "pori": float(scores[1]),
        "macchie": float(scores[2]),
        "occhiaie": float(scores[3]),
        "glow": float(scores[4]),
        "acne": float(scores[5]),
    }
    
    if len(scores) > 6:
        final_scores["pelle_pulita_percent"] = float(scores[6])
    
    print(f"✅ Score finali:")
    for key, value in final_scores.items():
        print(f"   - {key}: {value:.2f}")
        
except Exception as e:
    print(f"❌ Errore parsing: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ TUTTI I TEST PASSATI!")
print("=" * 60)
print("\nIl backend dovrebbe funzionare correttamente con H5.")
print("Puoi ora deployare su Railway!")
