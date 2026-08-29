# audit_siamese_results.py
#
# SIAMESE BiLSTM PERFORMANCE + ERROR AUDIT
#
# IMPORTANT:
# The original .keras checkpoint contains Lambda layers.
# Keras 3 may fail to deserialize those Lambda layers.
#
# This script therefore:
#   1. Rebuilds the original architecture manually.
#   2. Opens the .keras file as a ZIP archive.
#   3. Extracts model.weights.h5.
#   4. Loads ONLY the trained weights.
#   5. Performs the audit without deserializing the broken Lambda layers.
#
# This allows us to audit the EXISTING trained model.
#
# Seed: 42
# Architecture: exactly matches the original training script.
# ============================================================

import os
import random
import zipfile
import tempfile
import shutil

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Lambda
)

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BASE_DIR = r"processed\sequence_dataset\normalized"

MODEL_FILE = (
    r"processed\siamese_bilstm"
    r"\siamese_bilstm_best.keras"
)

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "train_normalized.npz"
)

VAL_FILE = os.path.join(
    BASE_DIR,
    "validation_normalized.npz"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "test_normalized.npz"
)

SEQ_LEN = 50
N_FEATURES = 16

EMBEDDING_DIM = 128

VAL_PAIRS_PER_USER = 8
TEST_PAIRS_PER_USER = 8

NEGATIVE_RATIO = 1.0

BATCH_SIZE = 128

# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("SIAMESE BiLSTM PERFORMANCE + ERROR AUDIT")
print("=" * 70)

print(f"Model : {MODEL_FILE}")
print(f"Seed  : {SEED}")

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

print(
    f"Train : X={X_train.shape}, "
    f"users={len(np.unique(pid_train))}"
)

print(
    f"Val   : X={X_val.shape}, "
    f"users={len(np.unique(pid_val))}"
)

print(
    f"Test  : X={X_test.shape}, "
    f"users={len(np.unique(pid_test))}"
)

# ============================================================
# DATA INTEGRITY AUDIT
# ============================================================

print("\n" + "=" * 70)
print("DATA INTEGRITY AUDIT")
print("=" * 70)


def audit_dataset(name, X, participant_id):

    print(f"\n{name}")

    print(f"  Shape       : {X.shape}")
    print(f"  Dtype       : {X.dtype}")
    print(f"  NaN count   : {np.isnan(X).sum():,}")
    print(f"  Inf count   : {np.isinf(X).sum():,}")
    print(
        f"  Participants: "
        f"{len(np.unique(participant_id)):,}"
    )

    assert X.ndim == 3
    assert X.shape[1] == SEQ_LEN
    assert X.shape[2] == N_FEATURES

    assert not np.isnan(X).any()
    assert not np.isinf(X).any()


audit_dataset(
    "TRAIN",
    X_train,
    pid_train
)

audit_dataset(
    "VALIDATION",
    X_val,
    pid_val
)

audit_dataset(
    "TEST",
    X_test,
    pid_test
)

# ============================================================
# PARTICIPANT LEAKAGE AUDIT
# ============================================================

print("\n" + "=" * 70)
print("PARTICIPANT LEAKAGE AUDIT")
print("=" * 70)

train_users = set(pid_train.tolist())
val_users = set(pid_val.tolist())
test_users = set(pid_test.tolist())

train_val_overlap = train_users & val_users
train_test_overlap = train_users & test_users
val_test_overlap = val_users & test_users

print(
    f"Train ∩ Validation : "
    f"{len(train_val_overlap)}"
)

print(
    f"Train ∩ Test       : "
    f"{len(train_test_overlap)}"
)

print(
    f"Validation ∩ Test  : "
    f"{len(val_test_overlap)}"
)

if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):

    print("Participant leakage : PASS")

else:

    raise RuntimeError(
        "Participant leakage detected."
    )

# ============================================================
# GROUP BY PARTICIPANT
# ============================================================

def group_by_participant(X, participant_ids):

    groups = {}

    for i, pid in enumerate(participant_ids):

        pid = int(pid)

        if pid not in groups:
            groups[pid] = []

        groups[pid].append(i)

    return groups


val_groups = group_by_participant(
    X_val,
    pid_val
)

test_groups = group_by_participant(
    X_test,
    pid_test
)

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

    users = np.array(
        list(groups.keys()),
        dtype=np.int32
    )

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

            i, j = rng.choice(
                indices,
                size=2,
                replace=False
            )

            left.append(X[i])
            right.append(X[j])
            labels.append(1.0)

    genuine_count = len(labels)

    # --------------------------------------------------------
    # IMPOSTOR PAIRS
    # --------------------------------------------------------

    target_impostors = int(
        genuine_count * negative_ratio
    )

    for _ in range(target_impostors):

        pid_a, pid_b = rng.choice(
            users,
            size=2,
            replace=False
        )

        i = rng.choice(
            groups[int(pid_a)]
        )

        j = rng.choice(
            groups[int(pid_b)]
        )

        left.append(X[i])
        right.append(X[j])
        labels.append(0.0)

    left = np.asarray(
        left,
        dtype=np.float32
    )

    right = np.asarray(
        right,
        dtype=np.float32
    )

    labels = np.asarray(
        labels,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------------

    order = rng.permutation(
        len(labels)
    )

    left = left[order]
    right = right[order]
    labels = labels[order]

    return left, right, labels


print("\n" + "=" * 70)
print("GENERATING AUDIT PAIRS")
print("=" * 70)

print("\nGenerating validation pairs...")

X1_val, X2_val, y_val = generate_pairs(
    X_val,
    val_groups,
    VAL_PAIRS_PER_USER,
    NEGATIVE_RATIO,
    SEED + 1
)

print(
    f"Validation pairs : {len(y_val):,}"
)

print(
    f"  Genuine        : "
    f"{np.sum(y_val == 1):,}"
)

print(
    f"  Impostor       : "
    f"{np.sum(y_val == 0):,}"
)

print("\nGenerating test pairs...")

X1_test, X2_test, y_test = generate_pairs(
    X_test,
    test_groups,
    TEST_PAIRS_PER_USER,
    NEGATIVE_RATIO,
    SEED + 2
)

print(
    f"Test pairs       : {len(y_test):,}"
)

print(
    f"  Genuine        : "
    f"{np.sum(y_test == 1):,}"
)

print(
    f"  Impostor       : "
    f"{np.sum(y_test == 0):,}"
)

# ============================================================
# REBUILD EXACT TRAINING ARCHITECTURE
# ============================================================

print("\n" + "=" * 70)
print("REBUILDING TRAINING ARCHITECTURE")
print("=" * 70)


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

    # EXACT SAME OPERATION AS ORIGINAL MODEL
    embeddings = Lambda(
        lambda z: tf.math.l2_normalize(
            z,
            axis=1
        ),
        output_shape=(EMBEDDING_DIM,),
        name="l2_normalized_embedding"
    )(x)

    return Model(
        inputs,
        embeddings,
        name="behavioral_bilstm_encoder"
    )


encoder = build_encoder()

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
    output_shape=(1,),
    name="euclidean_distance"
)(
    [embedding_a, embedding_b]
)

siamese = Model(
    [input_a, input_b],
    distance,
    name="siamese_bilstm"
)

print("\nArchitecture rebuilt successfully.")

print(
    f"Total parameters : "
    f"{siamese.count_params():,}"
)

# ============================================================
# EXTRACT WEIGHTS FROM .KERAS WITHOUT DESERIALIZING MODEL
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING TRAINED WEIGHTS")
print("=" * 70)

if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"Model checkpoint not found:\n{MODEL_FILE}"
    )

temp_dir = tempfile.mkdtemp(
    prefix="siamese_audit_"
)

weights_file = None

try:

    print(
        f"Checkpoint found : {MODEL_FILE}"
    )

    with zipfile.ZipFile(
        MODEL_FILE,
        "r"
    ) as z:

        names = z.namelist()

        print(
            f"Checkpoint files : {len(names)}"
        )

        # Keras .keras archives normally contain:
        #
        # config.json
        # metadata.json
        # model.weights.h5
        #
        possible_weights = [
            name
            for name in names
            if name.endswith(
                "model.weights.h5"
            )
        ]

        if len(possible_weights) == 0:

            # Fallback: find any HDF5 weight file
            possible_weights = [
                name
                for name in names
                if name.endswith(".h5")
            ]

        if len(possible_weights) == 0:

            raise RuntimeError(
                "No HDF5 weight file was found "
                "inside the .keras checkpoint."
            )

        if len(possible_weights) > 1:

            print(
                "Multiple weight files found:"
            )

            for item in possible_weights:
                print(
                    f"  {item}"
                )

        weight_member = possible_weights[0]

        print(
            f"Weight file       : "
            f"{weight_member}"
        )

        weights_file = os.path.join(
            temp_dir,
            "model.weights.h5"
        )

        with z.open(weight_member) as source, \
             open(weights_file, "wb") as target:

            shutil.copyfileobj(
                source,
                target
            )

    print(
        "\nLoading extracted weights "
        "into rebuilt architecture..."
    )

    siamese.load_weights(
        weights_file
    )

    print(
        "Trained weights loaded successfully."
    )

except Exception as e:

    print(
        "\nERROR while loading trained weights."
    )

    print(
        f"\n{type(e).__name__}: {e}"
    )

    print(
        "\nThe architecture and checkpoint "
        "weights could not be matched."
    )

    raise

# ============================================================
# MODEL WEIGHT SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("MODEL WEIGHT SANITY CHECK")
print("=" * 70)

all_weights = siamese.get_weights()

weight_count = len(all_weights)

total_weight_values = sum(
    np.prod(w.shape)
    for w in all_weights
)

nan_weights = sum(
    np.isnan(w).sum()
    for w in all_weights
)

inf_weights = sum(
    np.isinf(w).sum()
    for w in all_weights
)

print(
    f"Weight tensors     : "
    f"{weight_count}"
)

print(
    f"Weight values      : "
    f"{total_weight_values:,}"
)

print(
    f"NaN weight values  : "
    f"{nan_weights:,}"
)

print(
    f"Inf weight values  : "
    f"{inf_weights:,}"
)

if nan_weights != 0 or inf_weights != 0:

    raise RuntimeError(
        "Invalid NaN/Inf values detected "
        "inside trained weights."
    )

print("Weight integrity   : PASS")

# ============================================================
# GENERATE DISTANCES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING VALIDATION DISTANCES")
print("=" * 70)

val_distances = siamese.predict(
    [X1_val, X2_val],
    batch_size=BATCH_SIZE,
    verbose=1
).reshape(-1)

print(
    f"Validation distances: "
    f"{len(val_distances):,}"
)

print("\n" + "=" * 70)
print("GENERATING TEST DISTANCES")
print("=" * 70)

test_distances = siamese.predict(
    [X1_test, X2_test],
    batch_size=BATCH_SIZE,
    verbose=1
).reshape(-1)

print(
    f"Test distances: "
    f"{len(test_distances):,}"
)

# ============================================================
# DISTANCE SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("DISTANCE SANITY CHECK")
print("=" * 70)

print(
    f"Validation min : "
    f"{np.min(val_distances):.6f}"
)

print(
    f"Validation max : "
    f"{np.max(val_distances):.6f}"
)

print(
    f"Validation mean: "
    f"{np.mean(val_distances):.6f}"
)

print(
    f"Test min       : "
    f"{np.min(test_distances):.6f}"
)

print(
    f"Test max       : "
    f"{np.max(test_distances):.6f}"
)

print(
    f"Test mean      : "
    f"{np.mean(test_distances):.6f}"
)

if (
    not np.isfinite(val_distances).all()
    or not np.isfinite(test_distances).all()
):

    raise RuntimeError(
        "NaN/Inf detected in model distances."
    )

print("Distance integrity : PASS")

# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    distances,
    labels
):

    # Smaller distance = more genuine.
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

    distance_threshold = (
        -thresholds[eer_index]
    )

    return (
        auc,
        eer,
        distance_threshold
    )

# ============================================================
# VALIDATION METRICS
# ============================================================

val_auc, val_eer, threshold = calculate_metrics(
    val_distances,
    y_val
)

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(
    f"ROC-AUC        : "
    f"{val_auc:.6f}"
)

print(
    f"EER            : "
    f"{val_eer:.6f}"
)

print(
    f"EER threshold  : "
    f"{threshold:.6f}"
)

# ============================================================
# TEST METRICS
# ============================================================

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

# ============================================================
# FAR / FRR
# ============================================================

genuine = (
    y_test == 1
)

impostor = (
    y_test == 0
)

false_rejects = np.sum(
    (test_predictions == 0)
    & genuine
)

false_accepts = np.sum(
    (test_predictions == 1)
    & impostor
)

genuine_count = np.sum(
    genuine
)

impostor_count = np.sum(
    impostor
)

far = (
    false_accepts / impostor_count
    if impostor_count > 0
    else 0.0
)

frr = (
    false_rejects / genuine_count
    if genuine_count > 0
    else 0.0
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

# ============================================================
# SEPARATION AUDIT
# ============================================================

genuine_distances = test_distances[
    genuine
]

impostor_distances = test_distances[
    impostor
]

print("\n" + "=" * 70)
print("DISTANCE SEPARATION AUDIT")
print("=" * 70)

print(
    f"Genuine mean distance  : "
    f"{np.mean(genuine_distances):.6f}"
)

print(
    f"Genuine median distance: "
    f"{np.median(genuine_distances):.6f}"
)

print(
    f"Impostor mean distance : "
    f"{np.mean(impostor_distances):.6f}"
)

print(
    f"Impostor median distance: "
    f"{np.median(impostor_distances):.6f}"
)

print(
    f"Threshold              : "
    f"{threshold:.6f}"
)

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"ROC-AUC             : "
    f"{test_auc:.6f}"
)

print(
    f"EER                 : "
    f"{test_eer:.6f}"
)

print(
    f"Verification Acc.   : "
    f"{test_accuracy:.6f}"
)

print(
    f"FAR                 : "
    f"{far:.6f}"
)

print(
    f"FRR                 : "
    f"{frr:.6f}"
)

print(
    f"Validation Threshold: "
    f"{threshold:.6f}"
)

print("\nConfusion Matrix:")
print(
    "                 Predicted"
)
print(
    "              Impostor Genuine"
)
print(
    f"Actual Impostor "
    f"{tn:8d} {fp:8d}"
)
print(
    f"Actual Genuine  "
    f"{fn:8d} {tp:8d}"
)

# ============================================================
# FINAL AUDIT STATUS
# ============================================================

print("\n" + "=" * 70)
print("AUDIT STATUS")
print("=" * 70)

print("Dataset integrity       : PASS")
print("Participant separation  : PASS")
print("Weight integrity        : PASS")
print("Distance integrity      : PASS")
print("Model inference         : PASS")
print("Validation threshold    : PASS")
print("Final evaluation        : PASS")

print("\n" + "=" * 70)
print("SIAMESE BiLSTM AUDIT COMPLETE")
print("=" * 70)

print(
    "\nThe existing trained checkpoint was "
    "audited without deserializing its "
    "Lambda-based model configuration."
)

finally_cleanup = True

if finally_cleanup:

    try:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )
    except Exception:
        pass