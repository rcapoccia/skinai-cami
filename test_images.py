#!/usr/bin/env python3
"""
Test del modello con immagini reali di pelle
"""

import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

print("=" * 80)
print("TEST MODELLO ML - ANALISI IMMAGINI DI PELLE")
print("=" * 80)

# Carica modello
model_path = "skinai_ensemble_final.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Definisci immagini di test con risultati attesi
test_images = [
    {
        "path": "/home/ubuntu/upload/search_images/V7RAxBS9No2y.jpg",
        "name": "Pelle giovane perfetta",
        "expected": {
            "rughe": "BASSO (0-2)",
            "pori": "BASSO (0-3)",
            "macchie": "BASSO (0-2)",
            "occhiaie": "BASSO (0-1)",
            "glow": "ALTO (7-10)",
            "acne": "BASSO (0-1)",
            "pelle_pulita": "ALTO (80-100%)"
        }
    },
    {
        "path": "/home/ubuntu/upload/search_images/l8Syv9P2ALfH.jpg",
        "name": "Pelle chiara giovane",
        "expected": {
            "rughe": "BASSO (0-2)",
            "pori": "BASSO (0-3)",
            "macchie": "BASSO (0-2)",
            "occhiaie": "BASSO (0-2)",
            "glow": "ALTO (7-10)",
            "acne": "BASSO (0-1)",
            "pelle_pulita": "ALTO (75-100%)"
        }
    },
    {
        "path": "/home/ubuntu/upload/search_images/E5EbUxJjJy9h.jpg",
        "name": "Pelle perfetta giovane",
        "expected": {
            "rughe": "BASSO (0-2)",
            "pori": "BASSO (0-2)",
            "macchie": "BASSO (0-1)",
            "occhiaie": "BASSO (0-1)",
            "glow": "ALTO (8-10)",
            "acne": "BASSO (0-1)",
            "pelle_pulita": "ALTO (90-100%)"
        }
    },
    {
        "path": "/home/ubuntu/upload/search_images/SVuqkyW7Gg0X.jpg",
        "name": "Pelle con acne",
        "expected": {
            "rughe": "MEDIO (2-4)",
            "pori": "MEDIO-ALTO (4-6)",
            "macchie": "ALTO (6-8)",
            "occhiaie": "BASSO (0-2)",
            "glow": "BASSO (2-4)",
            "acne": "ALTO (7-10)",
            "pelle_pulita": "BASSO (10-30%)"
        }
    },
    {
        "path": "/home/ubuntu/upload/search_images/h2ALzTTcEJ5c.jpg",
        "name": "Pelle acneica",
        "expected": {
            "rughe": "BASSO (0-2)",
            "pori": "MEDIO (3-5)",
            "macchie": "ALTO (6-8)",
            "occhiaie": "BASSO (0-2)",
            "glow": "BASSO (1-3)",
            "acne": "ALTO (8-10)",
            "pelle_pulita": "BASSO (5-25%)"
        }
    },
    {
        "path": "/home/ubuntu/upload/search_images/JS6BSyfCUz3a.jpg",
        "name": "Pelle con acne diffusa",
        "expected": {
            "rughe": "MEDIO (2-4)",
            "pori": "MEDIO-ALTO (4-6)",
            "macchie": "ALTO (7-9)",
            "occhiaie": "MEDIO (2-4)",
            "glow": "BASSO (1-3)",
            "acne": "ALTO (8-10)",
            "pelle_pulita": "BASSO (5-20%)"
        }
    },
    {
        "path": "/home/ubuntu/upload/search_images/STAU0n3ydYxh.jpeg",
        "name": "Pelle matura con rughe",
        "expected": {
            "rughe": "ALTO (7-10)",
            "pori": "MEDIO-ALTO (4-6)",
            "macchie": "MEDIO-ALTO (4-6)",
            "occhiaie": "MEDIO-ALTO (4-6)",
            "glow": "BASSO (2-4)",
            "acne": "BASSO (0-2)",
            "pelle_pulita": "MEDIO (40-60%)"
        }
    },
    {
        "path": "/home/ubuntu/upload/search_images/goc9CVN5Dz6O.jpg",
        "name": "Pelle con molte rughe",
        "expected": {
            "rughe": "ALTO (8-10)",
            "pori": "MEDIO-ALTO (4-6)",
            "macchie": "MEDIO (3-5)",
            "occhiaie": "MEDIO (3-5)",
            "glow": "BASSO (1-3)",
            "acne": "BASSO (0-1)",
            "pelle_pulita": "MEDIO (35-55%)"
        }
    }
]

results = []

for idx, test_img in enumerate(test_images, 1):
    img_path = test_img["path"]
    
    if not os.path.exists(img_path):
        print(f"\n[{idx}] ❌ {test_img['name']}: FILE NON TROVATO")
        continue
    
    try:
        # Carica e processa immagine
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)
        
        # Inferenza
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        predictions = output_data[0] if len(output_data.shape) > 1 else output_data
        scores = predictions.tolist() if isinstance(predictions, np.ndarray) else list(predictions)
        
        while len(scores) < 7:
            scores.append(0.0)
        scores = scores[:7]
        
        # Calcola risultati
        final_scores = {
            "rughe": float(scores[0] * 10),
            "pori": float(scores[1] * 10),
            "macchie": float(scores[2] * 10),
            "occhiaie": float(scores[3] * 10),
            "glow": float(scores[4] * 10),
            "acne": float(scores[5] * 10),
            "pelle_pulita_percent": float(scores[6] * 100)
        }
        
        print(f"\n[{idx}] ✅ {test_img['name']}")
        print(f"    Risultati ATTESI:")
        for key, expected in test_img['expected'].items():
            print(f"      - {key}: {expected}")
        
        print(f"    Risultati OTTENUTI:")
        for key, value in final_scores.items():
            if key == "pelle_pulita_percent":
                print(f"      - {key}: {value:.1f}%")
            else:
                print(f"      - {key}: {value:.2f}")
        
        results.append({
            "name": test_img['name'],
            "scores": final_scores,
            "expected": test_img['expected']
        })
        
    except Exception as e:
        print(f"\n[{idx}] ❌ {test_img['name']}: ERRORE - {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Test completati: {len(results)}/{len(test_images)}")
print("\nAnalizza i risultati e verifica se sono coerenti con le aspettative!")
