import torch
from pathlib import Path

CHECKPOINT = Path(
    r"processed\siamese_bilstm_v3\model_training_v3"
    r"\checkpoints\best_siamese_bilstm_v3.pt"
)

print("=" * 70)
print("V3 CHECKPOINT ARCHITECTURE INSPECTOR")
print("=" * 70)

print(f"\nCheckpoint:\n{CHECKPOINT}")

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

print("\nCheckpoint object type:")
print(type(checkpoint))

# ---------------------------------------------------------
# Identify state dictionary
# ---------------------------------------------------------

state_dict = None

if isinstance(checkpoint, dict):

    possible_keys = [
        "model_state_dict",
        "state_dict",
        "model",
        "weights"
    ]

    for key in possible_keys:
        if key in checkpoint:
            candidate = checkpoint[key]

            if isinstance(candidate, dict):
                state_dict = candidate
                print(f"\nState dictionary found under key: {key}")
                break

    if state_dict is None:
        # Sometimes checkpoint itself is the state dict
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            state_dict = checkpoint
            print("\nCheckpoint itself is the state dictionary.")

if state_dict is None:
    raise RuntimeError(
        "Could not identify model state_dict in checkpoint."
    )

# ---------------------------------------------------------
# Print checkpoint metadata
# ---------------------------------------------------------

if isinstance(checkpoint, dict):

    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        if key not in [
            "model_state_dict",
            "state_dict",
            "model",
            "weights"
        ]:
            value = checkpoint[key]

            if isinstance(value, (str, int, float, bool)):
                print(f"  {key}: {value}")

# ---------------------------------------------------------
# Print exact architecture tensors
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL STATE DICTIONARY")
print("=" * 70)

total_parameters = 0

for name, tensor in state_dict.items():

    if not torch.is_tensor(tensor):
        print(f"{name}: {type(tensor)}")
        continue

    shape = tuple(tensor.shape)
    params = tensor.numel()

    total_parameters += params

    print(
        f"{name:<50} "
        f"shape={str(shape):<25} "
        f"params={params}"
    )

print("\n" + "=" * 70)
print("TOTAL PARAMETERS")
print("=" * 70)

print(f"{total_parameters:,}")

# ---------------------------------------------------------
# Important architecture clues
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ARCHITECTURE CLUES")
print("=" * 70)

for name, tensor in state_dict.items():

    if not torch.is_tensor(tensor):
        continue

    if name.startswith("embedding"):
        print(f"Embedding : {name} -> {tuple(tensor.shape)}")

    elif name.startswith("bilstm"):
        print(f"BiLSTM    : {name} -> {tuple(tensor.shape)}")

    elif (
        name.startswith("classifier")
        or name.startswith("fc")
        or name.startswith("head")
        or name.startswith("output")
        or name.startswith("projection")
    ):
        print(f"Head      : {name} -> {tuple(tensor.shape)}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)