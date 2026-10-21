# train_siamese_bilstm_v1.py

import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BASE_DIR = r"processed\sequence_dataset\normalized"
OUTPUT_DIR = r"processed\siamese_bilstm"

TRAIN_FILE = os.path.join(BASE_DIR, "train_normalized.npz")
VAL_FILE   = os.path.join(BASE_DIR, "validation_normalized.npz")
TEST_FILE  = os.path.join(BASE_DIR, "test_normalized.npz")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 50
N_FEATURES = 16

EMBEDDING_DIM = 128

BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 1e-3

TRAIN_PAIRS_PER_USER = 8
VAL_PAIRS_PER_USER = 8
TEST_PAIRS_PER_USER = 8

NEGATIVE_RATIO = 1.0

# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("SIAMESE BiLSTM — BASELINE TRAINING")
print("=" * 70)
print(f"Seed              : {SEED}")
print(f"Sequence length   : {SEQ_LEN}")
print(f"Features           : {N_FEATURES}")
print(f"Embedding          : {EMBEDDING_DIM}")
print(f"Batch size         : {BATCH_SIZE}")
print(f"Epochs             : {EPOCHS}")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path):
    data = np.load(path)

    X = data["X"].astype(np.float32)
    participant_id = data["participant_id"].astype(np.int32)

    return X, participant_id


print("\nLoading datasets...")

X_train, pid_train = load_dataset(TRAIN_FILE)
X_val, pid_val = load_dataset(VAL_FILE)
X_test, pid_test = load_dataset(TEST_FILE)

print(f"Train : X={X_train.shape}, users={len(np.unique(pid_train))}")
print(f"Val   : X={X_val.shape}, users={len(np.unique(pid_val))}")
print(f"Test  : X={X_test.shape}, users={len(np.unique(pid_test))}")


# ============================================================
# GROUP SEQUENCES BY PARTICIPANT
# ============================================================

def group_by_participant(X, participant_ids):

    groups = {}

    for i, pid in enumerate(participant_ids):
        pid = int(pid)

        if pid not in groups:
            groups[pid] = []

        groups[pid].append(i)

    return groups


train_groups = group_by_participant(X_train, pid_train)
val_groups = group_by_participant(X_val, pid_val)
test_groups = group_by_participant(X_test, pid_test)


# ============================================================
# PAIR GENERATION
# ============================================================

def generate_pairs(
    X,
    groups,
    pairs_per_user,
    negative_ratio=1.0,
    seed=42
):

    rng = np.random.default_rng(seed)

    users = np.array(list(groups.keys()))

    left = []
    right = []
    labels = []

    # --------------------------------------------------------
    # GENUINE PAIRS
    # --------------------------------------------------------

    for pid in users:

        indices = groups[int(pid)]

        if len(indices) < 2:
            continue

        for _ in range(pairs_per_user):

            i, j = rng.choice(indices, size=2, replace=False)

            left.append(X[i])
            right.append(X[j])
            labels.append(1.0)

    genuine_count = len(labels)

    # --------------------------------------------------------
    # IMPOSTOR PAIRS
    # --------------------------------------------------------

    target_impostors = int(genuine_count * negative_ratio)

    for _ in range(target_impostors):

        pid_a, pid_b = rng.choice(users, size=2, replace=False)

        i = rng.choice(groups[int(pid_a)])
        j = rng.choice(groups[int(pid_b)])

        left.append(X[i])
        right.append(X[j])
        labels.append(0.0)

    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    # --------------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------------

    order = rng.permutation(len(labels))

    left = left[order]
    right = right[order]
    labels = labels[order]

    return left, right, labels


print("\nGenerating training pairs...")

X1_train, X2_train, y_train = generate_pairs(
    X_train,
    train_groups,
    TRAIN_PAIRS_PER_USER,
    NEGATIVE_RATIO,
    SEED
)

print(f"Training pairs : {len(y_train):,}")
print(f"Genuine        : {np.sum(y_train == 1):,}")
print(f"Impostor       : {np.sum(y_train == 0):,}")


print("\nGenerating validation pairs...")

X1_val, X2_val, y_val = generate_pairs(
    X_val,
    val_groups,
    VAL_PAIRS_PER_USER,
    NEGATIVE_RATIO,
    SEED + 1
)

print(f"Validation pairs : {len(y_val):,}")


print("\nGenerating test pairs...")

X1_test, X2_test, y_test = generate_pairs(
    X_test,
    test_groups,
    TEST_PAIRS_PER_USER,
    NEGATIVE_RATIO,
    SEED + 2
)

print(f"Test pairs : {len(y_test):,}")


# ============================================================
# SIAMESE BiLSTM ENCODER
# ============================================================

def build_encoder():

    inputs = Input(
        shape=(SEQ_LEN, N_FEATURES),
        name="keystroke_sequence"
    )

    x = Bidirectional(
        LSTM(
            128,
            return_sequences=True,
            dropout=0.20
        ),
        name="bilstm_1"
    )(inputs)

    x = Bidirectional(
        LSTM(
            64,
            return_sequences=False,
            dropout=0.20
        ),
        name="bilstm_2"
    )(x)

    x = Dense(
        EMBEDDING_DIM,
        activation="relu",
        name="embedding_dense"
    )(x)

    x = Dropout(
        0.20,
        name="embedding_dropout"
    )(x)

    # L2-normalized behavioral embedding
    embeddings = Lambda(
        lambda z: tf.math.l2_normalize(z, axis=1),
        name="l2_normalized_embedding"
    )(x)

    return Model(
        inputs,
        embeddings,
        name="behavioral_bilstm_encoder"
    )


encoder = build_encoder()

encoder.summary()


# ============================================================
# SIAMESE NETWORK
# ============================================================

input_a = Input(
    shape=(SEQ_LEN, N_FEATURES),
    name="sequence_a"
)

input_b = Input(
    shape=(SEQ_LEN, N_FEATURES),
    name="sequence_b"
)

embedding_a = encoder(input_a)
embedding_b = encoder(input_b)


def euclidean_distance(vectors):

    a, b = vectors

    return tf.sqrt(
        tf.maximum(
            tf.reduce_sum(
                tf.square(a - b),
                axis=1,
                keepdims=True
            ),
            tf.keras.backend.epsilon()
        )
    )


distance = Lambda(
    euclidean_distance,
    name="euclidean_distance"
)([embedding_a, embedding_b])


siamese = Model(
    [input_a, input_b],
    distance,
    name="siamese_bilstm"
)


# ============================================================
# CONTRASTIVE LOSS
# ============================================================

MARGIN = 1.0


def contrastive_loss(y_true, distance):

    y_true = tf.cast(
        tf.reshape(y_true, tf.shape(distance)),
        tf.float32
    )

    positive_loss = y_true * tf.square(distance)

    negative_loss = (
        (1.0 - y_true)
        * tf.square(
            tf.maximum(MARGIN - distance, 0.0)
        )
    )

    return tf.reduce_mean(
        positive_loss + negative_loss
    )


siamese.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=contrastive_loss
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = os.path.join(
    OUTPUT_DIR,
    "siamese_bilstm_best.keras"
)

callbacks = [

    ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        mode="min",
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        mode="min",
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

history = siamese.fit(
    [X1_train, X2_train],
    y_train,
    validation_data=(
        [X1_val, X2_val],
        y_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# DISTANCE PREDICTION
# ============================================================

print("\nGenerating validation distances...")

val_distances = siamese.predict(
    [X1_val, X2_val],
    batch_size=BATCH_SIZE,
    verbose=1
).reshape(-1)


print("Generating test distances...")

test_distances = siamese.predict(
    [X1_test, X2_test],
    batch_size=BATCH_SIZE,
    verbose=1
).reshape(-1)


# ============================================================
# ROC / EER
# ============================================================

def calculate_metrics(distances, labels):

    # Smaller distance = more likely genuine.
    scores = -distances

    auc = roc_auc_score(
        labels,
        scores
    )

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    fnr = 1.0 - tpr

    eer_index = np.nanargmin(
        np.abs(fpr - fnr)
    )

    eer = (
        fpr[eer_index]
        + fnr[eer_index]
    ) / 2.0

    # Convert score threshold to distance threshold
    distance_threshold = -thresholds[eer_index]

    return auc, eer, distance_threshold


# ============================================================
# VALIDATION THRESHOLD
# ============================================================

val_auc, val_eer, threshold = calculate_metrics(
    val_distances,
    y_val
)

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(f"ROC-AUC        : {val_auc:.6f}")
print(f"EER            : {val_eer:.6f}")
print(f"EER threshold  : {threshold:.6f}")


# ============================================================
# TEST EVALUATION
# ============================================================

# Genuine if distance <= threshold
test_predictions = (
    test_distances <= threshold
).astype(np.int32)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_auc, test_eer, _ = calculate_metrics(
    test_distances,
    y_test
)


# FAR / FRR

genuine = y_test == 1
impostor = y_test == 0

false_rejects = np.sum(
    (test_predictions == 0) & genuine
)

false_accepts = np.sum(
    (test_predictions == 1) & impostor
)

far = (
    false_accepts / np.sum(impostor)
    if np.sum(impostor) > 0
    else 0
)

frr = (
    false_rejects / np.sum(genuine)
    if np.sum(genuine) > 0
    else 0
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"ROC-AUC             : {test_auc:.6f}")
print(f"EER                 : {test_eer:.6f}")
print(f"Verification Acc.   : {test_accuracy:.6f}")
print(f"FAR                 : {far:.6f}")
print(f"FRR                 : {frr:.6f}")
print(f"Validation Threshold: {threshold:.6f}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"Model saved to:")
print(checkpoint_path)
