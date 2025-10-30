# -------------------------------
# Leaf Disease Prediction with GrabCut - Colab
# -------------------------------

!pip install tensorflow opencv-python --quiet

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from google.colab import files
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load Trained Model
# -------------------------------
model = load_model("leaf_disease_model.h5")
print("✅ Loaded leaf disease model successfully!")

# Class names (must match training order)
class_names = ['appleblackrot', 'applescabe', 'cherrypowderymildew','peachbacterialspot','potatoearlyblight','potatolateblight','tomatobacterialspot','cornrust']

# -------------------------------
# Step 2: Upload a single leaf image
# -------------------------------
print("📤 Upload a leaf image to predict disease:")
uploaded = files.upload()
img_path = next(iter(uploaded))

# -------------------------------
# Step 3: Read Image
# -------------------------------
img = cv2.imread(img_path)
if img is None:
    raise ValueError("❌ Could not read the uploaded image.")

# -------------------------------
# Step 4: Apply GrabCut for segmentation
# -------------------------------
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

# Initial rectangle around leaf (assumes leaf is roughly centered)
height, width = img.shape[:2]
rect = (10, 10, width-20, height-20)

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Convert mask to binary: sure and probable foreground as 1, background as 0
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img_segmented = img * mask2[:, :, np.newaxis]

# -------------------------------
# Step 5: Preprocess Image
# -------------------------------
img_size = 224
img_rgb = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (img_size, img_size))
img_array = np.expand_dims(img_resized, axis=0) / 255.0

# -------------------------------
# Step 6: Make Prediction
# -------------------------------
pred_probs = model.predict(img_array)[0]
pred_idx = np.argmax(pred_probs)
pred_class = class_names[pred_idx]
confidence = pred_probs[pred_idx]

# -------------------------------
# Step 7: Display Result
# -------------------------------
plt.figure(figsize=(6,6))
plt.imshow(img_rgb)
plt.axis('off')
plt.title(f"Prediction: {pred_class}\nConfidence: {confidence*100:.2f}%", fontsize=14)
plt.show()

print(f"✅ Predicted Class: {pred_class}")
print(f"📊 Confidence: {confidence*100:.2f}%")
