import numpy as np
from tensorflow.keras.models import load_model
import os

# Load Model
MODEL_PATH = 'models/malaria_fixed.h5'
try:
    if os.path.exists(MODEL_PATH):
        print(f"Loading {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
    else:
        print("Model not found!")
        exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def predict_on(name, img_array):
    pred = model.predict(img_array, verbose=0)
    idx = np.argmax(pred[0])
    conf = pred[0][idx]
    label = "Uninfected" if idx == 0 else "Parasitized"
    print(f"Input: {name:15} -> Prediction: {label} ({conf:.2%})")

# 1. Black Image (Zeros)
black_img = np.zeros((1, 36, 36, 3), dtype='float32')
predict_on("Black (0s)", black_img)

# 2. White Image (1s)
white_img = np.ones((1, 36, 36, 3), dtype='float32')
predict_on("White (1s)", white_img)

# 3. Random Noise
noise_img = np.random.rand(1, 36, 36, 3).astype('float32')
predict_on("Random Noise", noise_img)
