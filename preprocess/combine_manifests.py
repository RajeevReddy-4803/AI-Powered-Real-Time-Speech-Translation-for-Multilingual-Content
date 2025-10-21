import pandas as pd
from pathlib import Path

# === Paths ===
base = Path(__file__).resolve().parents[1] / "data"
hindi_manifest = base / "hindi_preprocessed" / "train_manifest.csv"
english_manifest = base / "english_preprocessed" / "train_manifest.csv"
out_dir = base / "asr_combined"
out_dir.mkdir(exist_ok=True)

# === Load datasets ===
print("📂 Loading manifests...")
hi = pd.read_csv(hindi_manifest)
en = pd.read_csv(english_manifest)

print("Hindi:", len(hi), "rows | Missing transcripts:", hi['transcript'].isna().sum())
print("English:", len(en), "rows | Missing transcripts:", en['transcript'].isna().sum())

# === Drop NaN transcripts ===
hi = hi.dropna(subset=["transcript"])
en = en.dropna(subset=["transcript"])

# === Combine ===
combined = pd.concat([hi, en], ignore_index=True)
print(f"✅ Combined {len(combined)} rows total after dropping missing transcripts")

# === Save ===
combined_path = out_dir / "train_manifest.csv"
combined.to_csv(combined_path, index=False)

print(f"💾 Combined manifest saved to: {combined_path}")
print("Sample:\n", combined.head())
