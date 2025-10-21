# preprocess/split_combined.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

base = Path(__file__).resolve().parents[1] / "data" / "asr_combined"
df = pd.read_csv(base / "train_manifest.csv")

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)

train_df.to_csv(base / "train_manifest.csv", index=False)
val_df.to_csv(base / "val_manifest.csv", index=False)
test_df.to_csv(base / "test_manifest.csv", index=False)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
