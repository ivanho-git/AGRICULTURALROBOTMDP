# --------------------------------------------------
# 🌿 Leaf Disease Prediction with GrabCut Segmentation
# --------------------------------------------------
!pip install tensorflow opencv-python pillow numpy --quiet

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from google.colab import files
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# 1) Load Trained Model
# --------------------------------------------------
MODEL_PATH = "leaf_disease_model_final.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file '{MODEL_PATH}' not found. Please upload it first!")

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# --------------------------------------------------
# 2) Define Helper Functions
# --------------------------------------------------

def grabcut_segment(image):
    """Applies GrabCut to isolate the leaf from the background."""
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define a rectangle around the center region (where the leaf usually is)
    height, width = image.shape[:2]
    rect = (int(width * 0.05), int(height * 0.05),
            int(width * 0.9), int(height * 0.9))

    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    segmented = image * mask2[:, :, np.newaxis]
    return segmented

def preprocess_leaf(image_path, img_size=224):
    """Loads an image, applies GrabCut, preprocesses for MobileNetV2."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image file!")

    # Convert JFIF → RGB using PIL (OpenCV may fail to read JFIF correctly)
    if image_path.lower().endswith(".jfif"):
        pil_img = Image.open(image_path).convert("RGB")
        img = np.array(pil_img)[:, :, ::-1]  # Convert RGB → BGR for OpenCV

    # Resize for GrabCut performance
    img = cv2.resize(img, (300, 300))

    # Apply GrabCut segmentation
    leaf = grabcut_segment(img)

    # Resize to model input size and preprocess
    leaf_resized = cv2.resize(leaf, (img_size, img_size))
    leaf_array = preprocess_input(np.expand_dims(leaf_resized, axis=0).astype("float32"))
    return leaf_array, leaf, img

def display_images(original, segmented):
    """Displays original and segmented images side-by-side."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    ax[1].set_title("After GrabCut Segmentation")
    ax[1].axis("off")
    plt.show()

# --------------------------------------------------
# 3) Upload Image and Predict
# --------------------------------------------------
print("📸 Upload a leaf image (.jpg/.png/.jfif):")
uploaded = files.upload()
file_path = next(iter(uploaded))

# Preprocess
input_array, segmented_leaf, original_img = preprocess_leaf(file_path)

# Display segmentation
display_images(original_img, segmented_leaf)

# Predict
pred = model.predict(input_array)
confidence = np.max(pred)
predicted_class = np.argmax(pred)

# Try to infer class names if saved
try:
    with open("class_names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    label = class_names[predicted_class]
except:
    label = f"Class #{predicted_class}"

# --------------------------------------------------
# 4) Output Results
# --------------------------------------------------
print("\n🔍 Prediction Results:")
print(f"Predicted Disease: {label}")
print(f"Confidence: {confidence*100:.2f}%")
