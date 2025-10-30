# -------------------------------
# Leaf Disease Classification with GrabCut - Colab
# -------------------------------

!pip install tensorflow opencv-python --quiet

import os
import zipfile
import shutil
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import files

# -------------------------------
# Step 1: Upload the ZIP
# -------------------------------
print("📦 Upload your dataset ZIP (e.g., fin.zip)")
uploaded = files.upload()  # Upload fin.zip

zip_path = next(iter(uploaded))
extract_dir = "datasets"

# Unzip the uploaded file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f"✅ Extracted all files into '{extract_dir}'")

# -------------------------------
# Step 2: Fix extra top-level folder if exists
# -------------------------------
extracted_root = os.listdir(extract_dir)
if len(extracted_root) == 1 and os.path.isdir(os.path.join(extract_dir, extracted_root[0])):
    inner_folder = os.path.join(extract_dir, extracted_root[0])
    for folder_name in os.listdir(inner_folder):
        shutil.move(os.path.join(inner_folder, folder_name), extract_dir)
    shutil.rmtree(inner_folder)
    print("✅ Moved inner folders up one level")

# -------------------------------
# Step 3: Prepare dataset with GrabCut
# -------------------------------
categories = ['appleblackrot', 'applescabe', 'cherrypowderymildew','peachbactspot','potatoearlyblight','potatolateblight','tomatobacterialspot','cornrust']
img_size = 224
X, y = [], []

def apply_grabcut(img):
    """Apply GrabCut to segment the leaf from background."""
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    height, width = img.shape[:2]
    rect = (10, 10, width-20, height-20)  # Rough rectangle around leaf
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    return img * mask2[:, :, np.newaxis]

for label, category in enumerate(categories):
    folder = os.path.join(extract_dir, category)
    if not os.path.exists(folder):
        print(f"⚠️ Folder '{category}' not found, skipping.")
        continue

    count = 0
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif')):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is not None:
                # Apply GrabCut segmentation
                img_segmented = apply_grabcut(img)
                img_resized = cv2.resize(img_segmented, (img_size, img_size))
                X.append(img_resized)
                y.append(label)
                count += 1
    print(f"✅ Processed {count} images for class '{category}'")

if len(X) == 0:
    raise ValueError("❌ No images processed! Check folder structure and file extensions.")

X = np.array(X, dtype="float32") / 255.0
y = to_categorical(np.array(y), num_classes=len(categories))

# -------------------------------
# Step 4: Split data
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Step 5: Data augmentation
# -------------------------------
datagen = ImageDataGenerator(rotation_range=25, zoom_range=0.2,
                             width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, horizontal_flip=True)

# -------------------------------
# Step 6: Build Model
# -------------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(categories), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# Step 7: Train Model
# -------------------------------
epochs = 10
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    validation_data=(X_val, y_val),
    epochs=epochs
)

# -------------------------------
# Step 8: Save Model
# -------------------------------
model.save("leaf_disease_model.h5")
print("✅ Model saved as leaf_disease_model.h5")
files.download("leaf_disease_model.h5")
