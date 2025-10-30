# --------------------------------------------------
# Leaf Disease Classification (Optimized Transfer Learning + JFIF Support)
# --------------------------------------------------
!pip install tensorflow opencv-python scikit-learn pillow --quiet

import os, zipfile, shutil
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from google.colab import files

# --- Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
LEARNING_RATE_HEAD = 0.001
LEARNING_RATE_FT = 0.0001
EPOCHS_HEAD = 5
EPOCHS_FT = 15

# --------------------------------------------------
# 1) Upload and Setup Data Directory (with .jfif handling)
# --------------------------------------------------
print("📦 Upload your dataset ZIP (supports .jfif, .jpg, .png, etc.)")
try:
    uploaded = files.upload()
    zip_path = next(iter(uploaded))
    extract_dir = "datasets"

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    print(f"✅ Extracted all files into '{extract_dir}'")

    # Fix single top-level folder if present
    extracted_root = os.listdir(extract_dir)
    if len(extracted_root) == 1 and os.path.isdir(os.path.join(extract_dir, extracted_root[0])):
        inner = os.path.join(extract_dir, extracted_root[0])
        for name in os.listdir(inner):
            shutil.move(os.path.join(inner, name), extract_dir)
        shutil.rmtree(inner)
        print("Moved inner folders up one level")

    # ✅ Convert .jfif to .jpg for compatibility with keras dataset loader
    print("\n🧩 Converting .jfif files to .jpg ...")
    converted = 0
    for root, dirs, files_in_dir in os.walk(extract_dir):
        for fname in files_in_dir:
            if fname.lower().endswith(".jfif"):
                jfif_path = os.path.join(root, fname)
                jpg_path = jfif_path.rsplit(".", 1)[0] + ".jpg"
                try:
                    with Image.open(jfif_path) as img:
                        img.convert("RGB").save(jpg_path, "JPEG")
                    os.remove(jfif_path)
                    converted += 1
                except Exception as e:
                    print(f"⚠️ Failed to convert {fname}: {e}")
    print(f"✅ Converted {converted} .jfif files to .jpg\n")

    # Discover classes from directory names
    root_dir = extract_dir
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    print(f"✅ Detected classes ({len(class_names)} total): {class_names}")

    if len(class_names) < 2:
        raise ValueError("Expected multiple class folders, but found too few: {}".format(class_names))

except Exception as e:
    print(f"Error during upload/setup: {e}")
    exit()


# --------------------------------------------------
# 2) tf.data pipeline with MobileNetV2 preprocessing
# --------------------------------------------------
print("\n--- Setting up tf.data pipeline ---")

def preprocess_fn(image):
    """Scales input pixels from [0, 255] to [-1, 1] required by MobileNetV2."""
    return tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(image, tf.float32))

train_ds = tf.keras.utils.image_dataset_from_directory(
    root_dir,
    labels="inferred",
    class_names=class_names,
    label_mode="categorical",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    root_dir,
    labels="inferred",
    class_names=class_names,
    label_mode="categorical",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42,
)

train_ds = train_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)


# --------------------------------------------------
# 3) Data Augmentation
# --------------------------------------------------
aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

train_ds_aug = train_ds.map(lambda x, y: (aug(x), y), num_parallel_calls=AUTOTUNE)

# --------------------------------------------------
# 4) Build Model (Two-Stage Transfer Learning)
# --------------------------------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(len(class_names), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

base_model.trainable = False
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_HEAD), loss="categorical_crossentropy", metrics=["accuracy"])

print("\n--- Model Structure (Head Training Ready) ---")
model.summary()

cb_early = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy")
cb_lr    = ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_accuracy", verbose=1)

# --------------------------------------------------
# 5) Stage 1: Train Head
# --------------------------------------------------
print("\n--- Stage 1: Training Classifier Head ---")
history_head = model.fit(train_ds_aug, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=[cb_early, cb_lr])

# --------------------------------------------------
# 6) Stage 2: Fine-Tuning
# --------------------------------------------------
print("\n--- Stage 2: Fine-Tuning Backbone ---")
base_model.trainable = True
fine_tune_at = 120
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_FT), loss="categorical_crossentropy", metrics=["accuracy"])
history_ft = model.fit(train_ds_aug, validation_data=val_ds, epochs=EPOCHS_FT, callbacks=[cb_early, cb_lr])

# --------------------------------------------------
# 7) Final Evaluation
# --------------------------------------------------
print("\n--- Final Evaluation ---")
val_images, val_labels = [], []
for img, lbl in val_ds.unbatch():
    val_images.append(img.numpy())
    val_labels.append(lbl.numpy())

val_images = np.array(val_images)
val_labels = np.array(val_labels)
val_images_processed = preprocess_fn(tf.convert_to_tensor(val_images))

pred_probs = model.predict(val_images_processed, batch_size=BATCH_SIZE, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = np.argmax(val_labels, axis=1)

cm = confusion_matrix(true_labels, pred_labels)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names))

# --------------------------------------------------
# 8) Save Model
# --------------------------------------------------
model_filename = "leaf_disease_model_final.h5"
model.save(model_filename)
print(f"\n✅ Model saved as {model_filename}")
files.download(model_filename)
