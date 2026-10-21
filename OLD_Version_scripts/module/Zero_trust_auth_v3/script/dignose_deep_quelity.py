from pathlib import Path
import pandas as pd

path = Path("reports/deep_quality/full_deep_quality_report.csv")

df = pd.read_csv(path)

print("=" * 70)
print("DEEP QUALITY REJECTION DIAGNOSTIC")
print("=" * 70)

print("\nRows:", len(df))
print("Columns:")
for c in df.columns:
    print(" ", c)

print("\n" + "=" * 70)
print("REJECTION / STATUS DISTRIBUTION")
print("=" * 70)

for col in df.columns:
    if any(x in col.lower() for x in [
        "reason", "status", "eligible", "fail", "quality"
    ]):
        print(f"\n--- {col} ---")
        print(df[col].value_counts(dropna=False).head(30))

print("\n" + "=" * 70)
print("FIRST 20 ROWS")
print("=" * 70)

print(df.head(20).to_string(index=False))

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)