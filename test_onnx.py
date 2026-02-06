#!/usr/bin/env python3
"""
Test script per verificare il modello ONNX
"""

import os
import sys
import numpy as np
from PIL import Image
import onnxruntime as rt

print("=" * 60)
print("TEST ONNX MODEL - SKINGLOW AI")
print("=" * 60)

# Test 1: Verifica file modello
print("\n[TEST 1] Verifica file modello...")
model_path = "skin_analyzer.onnx"
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅ Modello trovato: {model_path} ({file_size:.2f} MB)")
else:
    print(f"❌ Modello NON trovato: {model_path}")
    sys.exit(1)

# Test 2: Carica il modello ONNX
print("\n[TEST 2] Caricamento modello ONNX...")
try:
    session = rt.InferenceSession(model_path)
    print("✅ Modello caricato con successo")
except Exception as e:
    print(f"❌ Errore caricamento modello: {e}")
    sys.exit(1)

# Test 3: Verifica input/output details
print("\n[TEST 3] Verifica input/output details...")
try:
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()
    
    print(f"✅ Input name: {input_info.name}")
    print(f"✅ Input shape: {input_info.shape}")
    print(f"✅ Input type: {input_info.type}")
    print(f"✅ Number of outputs: {len(output_info)}")
    for i, out in enumerate(output_info):
        print(f"   Output {i}: {out.name} - shape: {out.shape}")
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
    # Transpose to CHW format
    arr = np.transpose(arr, (2, 0, 1))
    # Add batch dimension
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
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: arr})
    print(f"✅ Inferenza completata")
    print(f"✅ Output shape: {output[0].shape}")
    print(f"✅ Output dtype: {output[0].dtype}")
except Exception as e:
    print(f"❌ Errore inferenza: {e}")
    sys.exit(1)

# Test 7: Parsing output
print("\n[TEST 7] Parsing output...")
try:
    predictions = output[0][0]  # Get first batch, all outputs
    
    print(f"✅ Numero di parametri: {len(predictions)}")
    print(f"✅ Valori grezzi: {predictions}")
    
    # Calcola i parametri finali
    final_scores = {
        "rughe": float(predictions[0]),
        "pori": float(predictions[1]),
        "macchie": float(predictions[2]),
        "occhiaie": float(predictions[3]),
        "disidratazione": float(predictions[4]),
        "acne": float(predictions[5])
    }
    
    print(f"✅ Score finali:")
    for key, value in final_scores.items():
        print(f"   - {key}: {value:.2f}/10")
        
except Exception as e:
    print(f"❌ Errore parsing: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ TUTTI I TEST PASSATI!")
print("=" * 60)
print("\nIl backend dovrebbe funzionare correttamente con ONNX.")
print("Puoi ora deployare su Railway!")
