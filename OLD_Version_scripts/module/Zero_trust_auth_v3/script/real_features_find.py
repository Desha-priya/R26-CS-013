#this runs after failing 150,000 more participate. 

from pathlib import Path
import pandas as pd

FEATURE_DIR = Path("processed/features")

files = list(FEATURE_DIR.glob("*_features.csv"))

file = files[0]

df = pd.read_csv(file)

print("=" * 70)
print("TEMPORAL STRUCTURE DIAGNOSTIC")
print("=" * 70)

print("\nFile:")
print(file)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 20 rows:")
print(
    df[
        [
            "PARTICIPANT_ID",
            "TEST_SECTION_ID",
            "KEYSTROKE_ID"
        ]
    ].head(20).to_string(index=False)
)

print("\nUnique sections:")
print(
    df["TEST_SECTION_ID"]
    .nunique()
)

print("\nFirst section:")
first_section = df["TEST_SECTION_ID"].iloc[0]

section = df[
    df["TEST_SECTION_ID"] == first_section
]

print(
    section[
        [
            "TEST_SECTION_ID",
            "KEYSTROKE_ID"
        ]
    ].head(50).to_string(index=False)
)

print("\nKEYSTROKE_ID statistics:")

print(
    pd.to_numeric(
        df["KEYSTROKE_ID"],
        errors="coerce"
    ).describe()
)

print("\nNumber of decreases globally:")

key_ids = pd.to_numeric(
    df["KEYSTROKE_ID"],
    errors="coerce"
).dropna().to_numpy()

if len(key_ids) > 1:
    print(
        int(
            (key_ids[1:] < key_ids[:-1]).sum()
        )
    )
else:
    print(0)

print("\n" + "=" * 70)