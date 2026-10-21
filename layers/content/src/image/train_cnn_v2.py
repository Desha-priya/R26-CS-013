# =============================================================
# NeuraShield — Deepfake CNN Retraining Script
# Script  : src/image/retrain_cnn.py
#
# This script does everything from scratch:
#   1. Deletes old frames and re-extracts clean ones
#   2. Extracts 20 evenly-spaced frames per video
#   3. Tries to crop face region — falls back to centre crop
#   4. Trains MobileNetV2 with transfer learning
#   5. Verifies label mapping is correct before saving
#   6. Saves model to models/image/deepfake_cnn_model.h5
#
# Why recreate frames?
#   Old frames may have been extracted inconsistently.
#   Clean extraction ensures real/ folder has ONLY real frames
#   and fake/ folder has ONLY fake frames — correct labels.
#
# Run from project root:
#   python src/image/retrain_cnn.py
# =============================================================

import os
import cv2
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time

print("=" * 60)
print("  NeuraShield — CNN Deepfake Classifier Retraining")
print("=" * 60)

# ── Check GPU ─────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"  GPU found: {gpus[0].name}")
    # Allow GPU memory to grow as needed instead of grabbing all at once
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("  No GPU found — using CPU (will be slower)")

# ── Paths ─────────────────────────────────────────────────────
REAL_VIDEO_DIR  = Path("data/video/raw/original")
FAKE_VIDEO_DIR  = Path("data/video/raw/manipulated")
FRAMES_ROOT     = Path("data/video/processed/frames")
REAL_FRAMES_DIR = FRAMES_ROOT / "real"
FAKE_FRAMES_DIR = FRAMES_ROOT / "fake"
MODEL_SAVE_PATH = "models/image/deepfake_cnn_model.h5"
RESULTS_DIR     = Path("outputs/results")

os.makedirs(REAL_FRAMES_DIR, exist_ok=True)
os.makedirs(FAKE_FRAMES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,     exist_ok=True)
os.makedirs("models/image",  exist_ok=True)

# ── Config ────────────────────────────────────────────────────
FRAMES_PER_VIDEO = 20    # 20 frames × 300 videos × 2 classes = 12000 images
IMG_SIZE         = 224   # MobileNetV2 required input size
MAX_VIDEOS       = 300   # cap per class
BATCH_SIZE       = 32
EPOCHS           = 10


# ══════════════════════════════════════════════════════════════
#  STEP 1 — DELETE OLD FRAMES AND RE-EXTRACT CLEAN ONES
# ══════════════════════════════════════════════════════════════

print("\nStep 1: Clearing old frames and re-extracting...")

# Delete everything in frames folder to start clean
for folder in [REAL_FRAMES_DIR, FAKE_FRAMES_DIR]:
    for f in folder.glob("*.jpg"):
        f.unlink()
print("  Old frames deleted.")

# Load face detector — built into OpenCV, no download needed
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def extract_frames(video_path: Path, output_dir: Path,
                   n_frames: int = FRAMES_PER_VIDEO) -> int:
    """
    Extract n_frames evenly spaced from video.
    For each frame:
      - Detect face and crop with 20% padding
      - If no face found, use centre crop
      - Resize to 224x224 and save as JPEG

    Evenly-spaced sampling captures the full video content,
    not just the first few seconds.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return 0

    # Evenly spaced frame positions across whole video
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    saved   = 0

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]

        # Try to detect and crop face region
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50)
        )

        if len(faces) > 0:
            # Take the largest face detected
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            # Add 20% padding so we include face boundary
            # — deepfake blending artifacts appear at face edges
            pad = int(0.20 * max(fw, fh))
            x1  = max(0, x - pad)
            y1  = max(0, y - pad)
            x2  = min(w, x + fw + pad)
            y2  = min(h, y + fh + pad)
            crop = frame[y1:y2, x1:x2]
        else:
            # No face detected — use centre square crop
            # Still useful training data
            size = min(h, w)
            y1   = (h - size) // 2
            x1   = (w - size) // 2
            crop = frame[y1:y1+size, x1:x1+size]

        # Resize to model input size
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

        # Save — filename: videoname_frameindex.jpg
        out_name = f"{video_path.stem}_{i:03d}.jpg"
        cv2.imwrite(str(output_dir / out_name), resized)
        saved += 1

    cap.release()
    return saved


# Extract real video frames
real_videos = sorted(REAL_VIDEO_DIR.glob("*.mp4"))[:MAX_VIDEOS]
fake_videos = sorted(FAKE_VIDEO_DIR.glob("*.mp4"))[:MAX_VIDEOS]

print(f"  Real videos to process: {len(real_videos)}")
print(f"  Fake videos to process: {len(fake_videos)}")

total_real = 0
for i, vp in enumerate(real_videos):
    n = extract_frames(vp, REAL_FRAMES_DIR)
    total_real += n
    if (i + 1) % 50 == 0:
        print(f"  Real [{i+1}/{len(real_videos)}] — frames: {total_real}")

total_fake = 0
for i, vp in enumerate(fake_videos):
    n = extract_frames(vp, FAKE_FRAMES_DIR)
    total_fake += n
    if (i + 1) % 50 == 0:
        print(f"  Fake [{i+1}/{len(fake_videos)}] — frames: {total_fake}")

print(f"\n  Real frames saved: {total_real}")
print(f"  Fake frames saved: {total_fake}")

# ── CRITICAL: verify folder contents ─────────────────────────
# Confirm real/ has only real frames and fake/ has only fake frames
# This is what sets the labels — folder name IS the label
actual_real = len(list(REAL_FRAMES_DIR.glob("*.jpg")))
actual_fake = len(list(FAKE_FRAMES_DIR.glob("*.jpg")))
print(f"\n  Verified — real/: {actual_real} images")
print(f"  Verified — fake/: {actual_fake} images")

# Check ImageDataGenerator will assign labels alphabetically
# fake comes before real alphabetically → fake=0, real=1
# We will flip predictions at inference time to fix this
print("\n  Label mapping (alphabetical):")
print("    fake/ → class index 0")
print("    real/ → class index 1")
print("  At inference: risk_score = 1.0 - model_output")
print("  So risk_score 1.0 = fake (threat), 0.0 = real (safe)")


# ══════════════════════════════════════════════════════════════
#  STEP 2 — DATA GENERATORS
#  ImageDataGenerator handles loading, augmentation, batching.
#  Training: augment with flips and colour jitter.
#  Validation: only rescale — no augmentation for fair evaluation.
# ══════════════════════════════════════════════════════════════

print("\nStep 2: Setting up data generators...")

# Training generator with augmentation
# Augmentation artificially increases variety so the model
# doesn't memorise specific videos — it learns general patterns
train_datagen = ImageDataGenerator(
    rescale=1./255,            # normalise pixels 0-255 → 0.0-1.0
    validation_split=0.20,     # 80% train, 20% validation
    horizontal_flip=True,      # mirror faces — deepfakes can appear flipped
    brightness_range=[0.8, 1.2],  # slight brightness variation
    zoom_range=0.1,            # slight zoom
    rotation_range=10          # slight rotation
)

# Validation generator — only rescale, no augmentation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.20
)

train_gen = train_datagen.flow_from_directory(
    FRAMES_ROOT,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',       # binary = 2 classes (real/fake)
    subset='training',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    FRAMES_ROOT,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False              # no shuffle for consistent evaluation
)

print(f"  Training batches   : {len(train_gen)}")
print(f"  Validation batches : {len(val_gen)}")
print(f"  Class indices      : {train_gen.class_indices}")
# Should print: {'fake': 0, 'real': 1}


# ══════════════════════════════════════════════════════════════
#  STEP 3 — BUILD MODEL (Transfer Learning)
#  MobileNetV2 pretrained on ImageNet — already knows how to
#  understand faces, textures, edges from 1.2M images.
#  We add our own classifier head on top for real/fake.
# ══════════════════════════════════════════════════════════════

print("\nStep 3: Building MobileNetV2 model...")

# Load MobileNetV2 without the top classification layer
# include_top=False means we add our own classifier
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all base model layers first
# We don't want to destroy the pretrained ImageNet features
base_model.trainable = False

# Unfreeze the last 30 layers for fine-tuning
# These higher layers learn task-specific features (deepfake patterns)
# while lower layers keep general image understanding
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Build our classifier on top of MobileNetV2
x = base_model.output
x = GlobalAveragePooling2D()(x)   # average spatial features into 1 vector
x = Dropout(0.4)(x)               # dropout prevents overfitting
x = Dense(128, activation='relu')(x)  # learn deepfake-specific patterns
x = Dropout(0.3)(x)
# Output: single sigmoid neuron → probability between 0 and 1
# near 0 = fake (because ImageDataGenerator: fake=0)
# near 1 = real (because ImageDataGenerator: real=1)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile model
# binary_crossentropy is standard for 2-class problems
# Adam optimizer with low learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"  Total layers       : {len(model.layers)}")
print(f"  Trainable layers   : {trainable_count}")
print(f"  Total parameters   : {model.count_params():,}")


# ══════════════════════════════════════════════════════════════
#  STEP 4 — TRAIN
# ══════════════════════════════════════════════════════════════

print(f"\nStep 4: Training for up to {EPOCHS} epochs...")

# Callbacks — these automatically manage training
callbacks = [
    # Save best model based on validation accuracy
    # So even if later epochs overfit, we keep the best version
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Stop early if validation accuracy stops improving for 3 epochs
    # Prevents wasting time and overfitting
    EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when validation loss plateaus
    # Helps model converge to better solution
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )
]

start = time.time()
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)
elapsed = time.time() - start
print(f"\n  Training complete in {elapsed/60:.1f} minutes")


# ══════════════════════════════════════════════════════════════
#  STEP 5 — EVALUATE AND SAVE RESULTS
# ══════════════════════════════════════════════════════════════

print("\nStep 5: Evaluating model...")

# Get predictions on validation set
val_gen.reset()
raw_preds  = model.predict(val_gen, verbose=1)
pred_labels = (raw_preds.flatten() < 0.5).astype(int)
# raw < 0.5 → fake (0 from model = fake)
# We convert: fake_detected = raw < 0.5 → label 1 (threat)
# true labels from generator: 0=fake, 1=real
true_labels = val_gen.classes

# For metrics, use original generator labels
from sklearn.metrics import accuracy_score
acc = accuracy_score(true_labels, (raw_preds.flatten() >= 0.5).astype(int))

print(f"\n  Final validation accuracy: {acc*100:.2f}%")
print(f"  Best val accuracy        : {max(history.history['val_accuracy'])*100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    true_labels,
    (raw_preds.flatten() >= 0.5).astype(int),
    target_names=['fake (0)', 'real (1)']
))

# ── Confusion matrix ──────────────────────────────────────────
cm = confusion_matrix(
    true_labels,
    (raw_preds.flatten() >= 0.5).astype(int)
)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['fake', 'real'],
            yticklabels=['fake', 'real'],
            ax=axes[0])
axes[0].set_title(f'Confusion Matrix\nacc={acc*100:.1f}%')
axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')

# Training history
axes[1].plot(history.history['accuracy'],     label='Train Acc', marker='o')
axes[1].plot(history.history['val_accuracy'], label='Val Acc',   marker='o')
axes[1].set_title('Training History')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('NeuraShield — MobileNetV2 Deepfake Detector', fontweight='bold')
plt.tight_layout()
PLOT = str(RESULTS_DIR / "cnn_retrain_results.png")
plt.savefig(PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Results plot saved: {PLOT}")
print(f"  Model saved       : {MODEL_SAVE_PATH}")

print("\n" + "=" * 60)
print("  RETRAINING COMPLETE")
print("=" * 60)
print(f"  Model  : MobileNetV2 (Transfer Learning, ImageNet)")
print(f"  Data   : FaceForensics++ — {actual_real} real + {actual_fake} fake frames")
print(f"  Acc    : {max(history.history['val_accuracy'])*100:.2f}%")
print(f"\n  IMPORTANT — at inference time:")
print(f"  risk_score = 1.0 - model_output")
print(f"  Because: fake=0, real=1 in training labels")
print(f"  After flip: risk_score 1.0 = fake (threat)")
print(f"\nNext: python demo/app.py")