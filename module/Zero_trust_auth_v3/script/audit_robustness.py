# ============================================================
# audit_robustness.py
#
# Robustness / Generalization Audit
# Siamese BiLSTM Behavioral Authentication
#
# IMPORTANT:
# - Does NOT retrain the model.
# - Does NOT deserialize the Lambda-based .keras model.
# - Rebuilds the original architecture and loads model.weights.h5
#   directly from the .keras archive.
# - Uses validation only for threshold selection.
# - Test set remains completely unseen for threshold selection.
# ============================================================

import os
import random
import zipfile
import tempfile
import shutil

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Lambda

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BASE_DIR = r"processed\sequence_dataset\normalized"
MODEL_FILE = r"processed\siamese_bilstm\siamese_bilstm_best.keras"

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

BATCH_SIZE = 128

# Must match the trained architecture
LSTM_1_UNITS = 128
LSTM_2_UNITS = 64

# Pair generation
PAIRS_PER_USER = 8
NEGATIVE_RATIO = 1.0

# Bootstrap
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 42

# Enrollment robustness
ENROLLMENT_LEVELS = [2, 3, 4, 5, 6, 8]

# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("SIAMESE BiLSTM ROBUSTNESS / GENERALIZATION AUDIT")
print("=" * 70)

print(f"Seed                : {SEED}")
print(f"Model               : {MODEL_FILE}")
print(f"Sequence length     : {SEQ_LEN}")
print(f"Features            : {N_FEATURES}")
print(f"Embedding dimension : {EMBEDDING_DIM}")
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
# DATA INTEGRITY
# ============================================================

print("\n" + "=" * 70)
print("DATA INTEGRITY")
print("=" * 70)

for name, X, pid in [
    ("TRAIN", X_train, pid_train),
    ("VALIDATION", X_val, pid_val),
    ("TEST", X_test, pid_test)
]:

    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())

    print(f"\n{name}")
    print(f"  Shape        : {X.shape}")
    print(f"  Dtype        : {X.dtype}")
    print(f"  NaN          : {nan_count}")
    print(f"  Inf          : {inf_count}")
    print(f"  Participants : {len(np.unique(pid))}")

    if nan_count != 0 or inf_count != 0:
        raise RuntimeError(
            f"{name} contains NaN or Inf values."
        )


# ============================================================
# PARTICIPANT LEAKAGE
# ============================================================

print("\n" + "=" * 70)
print("PARTICIPANT LEAKAGE")
print("=" * 70)

train_users = set(np.unique(pid_train))
val_users = set(np.unique(pid_val))
test_users = set(np.unique(pid_test))

train_val = train_users & val_users
train_test = train_users & test_users
val_test = val_users & test_users

print(f"Train ∩ Validation : {len(train_val)}")
print(f"Train ∩ Test       : {len(train_test)}")
print(f"Validation ∩ Test  : {len(val_test)}")

if train_val or train_test or val_test:
    raise RuntimeError(
        "Participant leakage detected."
    )

print("Participant separation : PASS")


# ============================================================
# REBUILD ORIGINAL SIAMESE ARCHITECTURE
# ============================================================

print("\n" + "=" * 70)
print("REBUILDING SIAMESE BiLSTM ARCHITECTURE")
print("=" * 70)


def build_encoder():

    inputs = Input(
        shape=(SEQ_LEN, N_FEATURES),
        name="keystroke_sequence"
    )

    x = Bidirectional(
        LSTM(
            LSTM_1_UNITS,
            return_sequences=True,
            dropout=0.20
        ),
        name="bilstm_1"
    )(inputs)

    x = Bidirectional(
        LSTM(
            LSTM_2_UNITS,
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

    embeddings = Lambda(
        lambda z: tf.math.l2_normalize(z, axis=1),
        output_shape=(EMBEDDING_DIM,),
        name="l2_normalized_embedding"
    )(x)

    return Model(
        inputs,
        embeddings,
        name="behavioral_bilstm_encoder"
    )


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

distance = Lambda(
    euclidean_distance,
    output_shape=(1,),
    name="euclidean_distance"
)(
    [embedding_a, embedding_b]
)

model = Model(
    [input_a, input_b],
    distance,
    name="siamese_bilstm"
)

print("Architecture rebuilt successfully.")
print(
    f"Total parameters : "
    f"{model.count_params():,}"
)


# ============================================================
# EXTRACT WEIGHTS FROM .KERAS ARCHIVE
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING TRAINED WEIGHTS")
print("=" * 70)

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_FILE}"
    )

temp_dir = tempfile.mkdtemp(
    prefix="siamese_audit_"
)

try:

    with zipfile.ZipFile(
        MODEL_FILE,
        "r"
    ) as z:

        members = z.namelist()

        weight_members = [
            m for m in members
            if m.endswith("model.weights.h5")
        ]

        if not weight_members:

            weight_members = [
                m for m in members
                if m.endswith(".weights.h5")
            ]

        if not weight_members:
            raise RuntimeError(
                "Could not find model weights inside .keras archive."
            )

        weight_member = weight_members[0]

        print(
            f"Weight file inside archive : "
            f"{weight_member}"
        )

        z.extract(
            weight_member,
            temp_dir
        )

        weight_path = os.path.join(
            temp_dir,
            weight_member
        )

    print("\nLoading trained weights...")

    model.load_weights(weight_path)

    print("Trained weights loaded successfully.")

    # ========================================================
    # WEIGHT SANITY
    # ========================================================

    print("\n" + "=" * 70)
    print("WEIGHT SANITY")
    print("=" * 70)

    total_values = 0
    nan_values = 0
    inf_values = 0
    tensors = 0

    for weight in model.weights:

        values = weight.numpy()

        tensors += 1
        total_values += values.size
        nan_values += np.isnan(values).sum()
        inf_values += np.isinf(values).sum()

    print(f"Weight tensors    : {tensors}")
    print(f"Weight values     : {total_values:,}")
    print(f"NaN weight values : {nan_values}")
    print(f"Inf weight values : {inf_values}")

    if nan_values or inf_values:
        raise RuntimeError(
            "Invalid values found in trained weights."
        )

    print("Weight integrity : PASS")


    # ========================================================
    # GROUP BY PARTICIPANT
    # ========================================================

    def group_by_participant(X, pids):

        groups = {}

        for i, pid in enumerate(pids):

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


    # ========================================================
    # PAIR GENERATION
    # ========================================================

    def generate_pairs(
        X,
        groups,
        pairs_per_user,
        negative_ratio,
        seed
    ):

        rng = np.random.default_rng(seed)

        users = np.array(
            list(groups.keys()),
            dtype=np.int32
        )

        left = []
        right = []
        labels = []

        # ------------------------------
        # Genuine
        # ------------------------------

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
                labels.append(1)

        genuine_count = len(labels)

        # ------------------------------
        # Impostor
        # ------------------------------

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
            labels.append(0)

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
            dtype=np.int32
        )

        order = rng.permutation(
            len(labels)
        )

        return (
            left[order],
            right[order],
            labels[order]
        )


    # ========================================================
    # VALIDATION / TEST PAIRS
    # ========================================================

    print("\n" + "=" * 70)
    print("GENERATING VALIDATION / TEST PAIRS")
    print("=" * 70)

    X1_val, X2_val, y_val = generate_pairs(
        X_val,
        val_groups,
        PAIRS_PER_USER,
        NEGATIVE_RATIO,
        SEED + 1
    )

    X1_test, X2_test, y_test = generate_pairs(
        X_test,
        test_groups,
        PAIRS_PER_USER,
        NEGATIVE_RATIO,
        SEED + 2
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


    # ========================================================
    # DISTANCES
    # ========================================================

    print("\n" + "=" * 70)
    print("GENERATING VALIDATION DISTANCES")
    print("=" * 70)

    val_distances = model.predict(
        [X1_val, X2_val],
        batch_size=BATCH_SIZE,
        verbose=1
    ).reshape(-1)

    print(
        f"Validation distances : "
        f"{len(val_distances):,}"
    )

    print("\n" + "=" * 70)
    print("GENERATING TEST DISTANCES")
    print("=" * 70)

    test_distances = model.predict(
        [X1_test, X2_test],
        batch_size=BATCH_SIZE,
        verbose=1
    ).reshape(-1)

    print(
        f"Test distances : "
        f"{len(test_distances):,}"
    )


    # ========================================================
    # THRESHOLD FROM VALIDATION
    # ========================================================

    def calculate_eer_threshold(
        distances,
        labels
    ):

        scores = -distances

        fpr, tpr, thresholds = roc_curve(
            labels,
            scores
        )

        fnr = 1.0 - tpr

        index = np.nanargmin(
            np.abs(fpr - fnr)
        )

        eer = (
            fpr[index] +
            fnr[index]
        ) / 2.0

        threshold = -thresholds[index]

        auc = roc_auc_score(
            labels,
            scores
        )

        return auc, eer, threshold


    val_auc, val_eer, threshold = calculate_eer_threshold(
        val_distances,
        y_val
    )

    print("\n" + "=" * 70)
    print("VALIDATION OPERATING POINT")
    print("=" * 70)

    print(f"ROC-AUC       : {val_auc:.6f}")
    print(f"EER           : {val_eer:.6f}")
    print(f"EER threshold : {threshold:.6f}")


    # ========================================================
    # TEST METRICS
    # ========================================================

    def calculate_test_metrics(
        distances,
        labels,
        threshold
    ):

        predictions = (
            distances <= threshold
        ).astype(np.int32)

        genuine = labels == 1
        impostor = labels == 0

        tp = np.sum(
            predictions[genuine] == 1
        )

        fn = np.sum(
            predictions[genuine] == 0
        )

        tn = np.sum(
            predictions[impostor] == 0
        )

        fp = np.sum(
            predictions[impostor] == 1
        )

        far = (
            fp / np.sum(impostor)
        )

        frr = (
            fn / np.sum(genuine)
        )

        accuracy = (
            (tp + tn) /
            len(labels)
        )

        auc = roc_auc_score(
            labels,
            -distances
        )

        return {
            "accuracy": accuracy,
            "far": far,
            "frr": frr,
            "auc": auc,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }


    test_metrics = calculate_test_metrics(
        test_distances,
        y_test,
        threshold
    )

    print("\n" + "=" * 70)
    print("BASELINE TEST RESULT")
    print("=" * 70)

    print(
        f"ROC-AUC  : "
        f"{test_metrics['auc']:.6f}"
    )

    print(
        f"Accuracy : "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"FAR      : "
        f"{test_metrics['far']:.6f}"
    )

    print(
        f"FRR      : "
        f"{test_metrics['frr']:.6f}"
    )


    # ========================================================
    # THRESHOLD SENSITIVITY
    # ========================================================

    print("\n" + "=" * 70)
    print("THRESHOLD SENSITIVITY AUDIT")
    print("=" * 70)

    thresholds_to_test = np.unique(
        np.round(
            np.concatenate(
                [
                    np.linspace(
                        max(0.05, threshold - 0.20),
                        threshold + 0.20,
                        17
                    ),
                    np.array([threshold])
                ]
            ),
            6
        )
    )

    print(
        "\nThreshold        FAR          FRR          Accuracy"
    )

    for t in thresholds_to_test:

        predictions = (
            test_distances <= t
        ).astype(np.int32)

        genuine = y_test == 1
        impostor = y_test == 0

        far = np.mean(
            predictions[impostor] == 1
        )

        frr = np.mean(
            predictions[genuine] == 0
        )

        accuracy = np.mean(
            predictions == y_test
        )

        marker = (
            " <-- validation EER threshold"
            if abs(t - threshold) < 1e-7
            else ""
        )

        print(
            f"{t:10.6f}    "
            f"{far:10.6f}    "
            f"{frr:10.6f}    "
            f"{accuracy:10.6f}"
            f"{marker}"
        )


    # ========================================================
    # PER-PARTICIPANT ROBUSTNESS
    # ========================================================

    print("\n" + "=" * 70)
    print("PER-PARTICIPANT ROBUSTNESS AUDIT")
    print("=" * 70)

    rng = np.random.default_rng(
        SEED + 100
    )

    user_results = []

    test_users = sorted(
        test_groups.keys()
    )

    for pid in test_users:

        indices = test_groups[pid]

        if len(indices) < 2:
            continue

        genuine_attempts = min(
            PAIRS_PER_USER,
            len(indices) * (len(indices) - 1) // 2
        )

        genuine_distances = []

        # deterministic genuine pairs
        for k in range(genuine_attempts):

            i, j = rng.choice(
                indices,
                size=2,
                replace=False
            )

            d = model.predict(
                [
                    X_test[i:i+1],
                    X_test[j:j+1]
                ],
                verbose=0
            )[0, 0]

            genuine_distances.append(
                float(d)
            )

        # ----------------------------------------------------
        # Impostor attempts against this user
        # ----------------------------------------------------

        other_users = [
            u for u in test_users
            if u != pid
        ]

        impostor_distances = []

        for k in range(
            PAIRS_PER_USER
        ):

            attacker = rng.choice(
                other_users
            )

            i = rng.choice(
                indices
            )

            j = rng.choice(
                test_groups[attacker]
            )

            d = model.predict(
                [
                    X_test[i:i+1],
                    X_test[j:j+1]
                ],
                verbose=0
            )[0, 0]

            impostor_distances.append(
                float(d)
            )

        genuine_distances = np.asarray(
            genuine_distances
        )

        impostor_distances = np.asarray(
            impostor_distances
        )

        genuine_acceptance = np.mean(
            genuine_distances <= threshold
        )

        impostor_rejection = np.mean(
            impostor_distances > threshold
        )

        user_results.append(
            (
                pid,
                len(indices),
                genuine_acceptance,
                impostor_rejection,
                np.mean(genuine_distances),
                np.mean(impostor_distances)
            )
        )

    user_results = np.asarray(
        user_results,
        dtype=object
    )

    genuine_rates = user_results[:, 2].astype(
        np.float64
    )

    impostor_rejection_rates = user_results[:, 3].astype(
        np.float64
    )

    print(
        f"Participants evaluated : "
        f"{len(user_results):,}"
    )

    print(
        f"\nGenuine acceptance rate:"
    )

    print(
        f"  Mean   : "
        f"{np.mean(genuine_rates):.6f}"
    )

    print(
        f"  Median : "
        f"{np.median(genuine_rates):.6f}"
    )

    print(
        f"  Min    : "
        f"{np.min(genuine_rates):.6f}"
    )

    print(
        f"  Max    : "
        f"{np.max(genuine_rates):.6f}"
    )

    print(
        f"\nImpostor rejection rate:"
    )

    print(
        f"  Mean   : "
        f"{np.mean(impostor_rejection_rates):.6f}"
    )

    print(
        f"  Median : "
        f"{np.median(impostor_rejection_rates):.6f}"
    )

    print(
        f"  Min    : "
        f"{np.min(impostor_rejection_rates):.6f}"
    )

    print(
        f"  Max    : "
        f"{np.max(impostor_rejection_rates):.6f}"
    )


    # ========================================================
    # WORST USERS
    # ========================================================

    print("\n" + "=" * 70)
    print("LOW-PERFORMING PARTICIPANT AUDIT")
    print("=" * 70)

    sorted_users = sorted(
        user_results,
        key=lambda x: x[2]
    )

    print(
        "\nWorst participants by genuine acceptance:"
    )

    print(
        "Participant    Sequences    "
        "GenuineAccept    ImpostorReject"
    )

    for row in sorted_users[:20]:

        print(
            f"{int(row[0]):10d}    "
            f"{int(row[1]):9d}    "
            f"{float(row[2]):13.4f}    "
            f"{float(row[3]):14.4f}"
        )


    # ========================================================
    # ENROLLMENT / DATA-VOLUME ROBUSTNESS
    #
    # This is NOT a retraining experiment.
    #
    # We progressively restrict each participant to the first
    # N available sequences and evaluate genuine verification
    # using those sequences.
    # ========================================================

    print("\n" + "=" * 70)
    print("DATA-VOLUME / ENROLLMENT ROBUSTNESS")
    print("=" * 70)

    print(
        "\nThis evaluates how much behavioral data is available "
        "per participant."
    )

    print(
        "\nSequences/user    Genuine acceptance"
    )

    enrollment_rng = np.random.default_rng(
        SEED + 500
    )

    for level in ENROLLMENT_LEVELS:

        rates = []

        for pid in test_users:

            indices = test_groups[pid]

            if len(indices) < level:
                continue

            selected = indices[:level]

            if len(selected) < 2:
                continue

            local_accepts = []

            for _ in range(
                min(8, level * (level - 1) // 2)
            ):

                i, j = enrollment_rng.choice(
                    selected,
                    size=2,
                    replace=False
                )

                d = model.predict(
                    [
                        X_test[i:i+1],
                        X_test[j:j+1]
                    ],
                    verbose=0
                )[0, 0]

                local_accepts.append(
                    d <= threshold
                )

            rates.append(
                np.mean(local_accepts)
            )

        if rates:

            print(
                f"{level:15d}    "
                f"{np.mean(rates):.6f}"
            )


    # ========================================================
    # BOOTSTRAP CONFIDENCE INTERVALS
    #
    # Bootstrap at PAIR level for descriptive uncertainty.
    #
    # We explicitly label these pair-level CIs.
    # ========================================================

    print("\n" + "=" * 70)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)

    bootstrap_rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    n = len(y_test)

    auc_values = []
    accuracy_values = []
    far_values = []
    frr_values = []

    print(
        f"Bootstrap iterations : "
        f"{BOOTSTRAP_ITERATIONS:,}"
    )

    for iteration in range(
        BOOTSTRAP_ITERATIONS
    ):

        sample_idx = bootstrap_rng.integers(
            0,
            n,
            size=n
        )

        sample_y = y_test[sample_idx]
        sample_d = test_distances[sample_idx]

        # AUC
        if (
            np.sum(sample_y == 1) == 0
            or
            np.sum(sample_y == 0) == 0
        ):
            continue

        sample_pred = (
            sample_d <= threshold
        ).astype(np.int32)

        auc_values.append(
            roc_auc_score(
                sample_y,
                -sample_d
            )
        )

        accuracy_values.append(
            np.mean(
                sample_pred == sample_y
            )
        )

        impostor = sample_y == 0
        genuine = sample_y == 1

        far_values.append(
            np.mean(
                sample_pred[impostor] == 1
            )
        )

        frr_values.append(
            np.mean(
                sample_pred[genuine] == 0
            )
        )

    def ci(values):

        return (
            np.percentile(values, 2.5),
            np.percentile(values, 97.5)
        )

    auc_low, auc_high = ci(
        np.asarray(auc_values)
    )

    acc_low, acc_high = ci(
        np.asarray(accuracy_values)
    )

    far_low, far_high = ci(
        np.asarray(far_values)
    )

    frr_low, frr_high = ci(
        np.asarray(frr_values)
    )

    print(
        "\n95% bootstrap confidence intervals:"
    )

    print(
        f"ROC-AUC   : "
        f"{test_metrics['auc']:.6f} "
        f"[{auc_low:.6f}, {auc_high:.6f}]"
    )

    print(
        f"Accuracy  : "
        f"{test_metrics['accuracy']:.6f} "
        f"[{acc_low:.6f}, {acc_high:.6f}]"
    )

    print(
        f"FAR       : "
        f"{test_metrics['far']:.6f} "
        f"[{far_low:.6f}, {far_high:.6f}]"
    )

    print(
        f"FRR       : "
        f"{test_metrics['frr']:.6f} "
        f"[{frr_low:.6f}, {frr_high:.6f}]"
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("ROBUSTNESS AUDIT SUMMARY")
    print("=" * 70)

    print(
        f"Test ROC-AUC          : "
        f"{test_metrics['auc']:.6f}"
    )

    print(
        f"Test Accuracy         : "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"Test FAR              : "
        f"{test_metrics['far']:.6f}"
    )

    print(
        f"Test FRR              : "
        f"{test_metrics['frr']:.6f}"
    )

    print(
        f"Validation threshold  : "
        f"{threshold:.6f}"
    )

    print(
        f"Participants evaluated: "
        f"{len(user_results):,}"
    )

    print(
        f"Bootstrap iterations   : "
        f"{len(auc_values):,}"
    )

    print("\n" + "=" * 70)
    print("ROBUSTNESS AUDIT COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "No model retraining was performed."
    )

    print(
        "No test-derived threshold was used."
    )

    print(
        "Validation was used for operating-threshold selection."
    )

    print(
        "The test set remained evaluation-only."
    )

finally:

    shutil.rmtree(
        temp_dir,
        ignore_errors=True
    )