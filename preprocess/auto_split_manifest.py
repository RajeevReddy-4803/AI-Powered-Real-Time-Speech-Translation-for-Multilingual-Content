# preprocess/auto_split_manifest.py
"""
Auto-split bilingual_manifest.csv into train/val/test.
Uses stratified split on duration bins to keep distribution balanced.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

def auto_split(manifest_csv, out_dir, val_frac=0.05, test_frac=0.05, min_dur=0.5, max_dur=20.0, random_state=42):
    p = Path(manifest_csv)
    if not p.exists():
        raise FileNotFoundError(manifest_csv)
    df = pd.read_csv(p)
    print(f"Loaded {len(df)} rows from {manifest_csv}")

    # filter durations
    if 'duration_sec' in df.columns:
        before = len(df)
        df = df[(df['duration_sec'] >= min_dur) & (df['duration_sec'] <= max_dur)]
        print(f"Filtered by duration [{min_dur}, {max_dur}] : {before} -> {len(df)}")

    # create duration bins for stratify
    if 'duration_sec' in df.columns:
        df['dur_bin'] = pd.qcut(df['duration_sec'].clip(lower=min_dur), q=5, duplicates='drop')

        train_df, rest = train_test_split(df, test_size=(val_frac + test_frac), random_state=random_state, stratify=df['dur_bin'])
        # proportionally split rest
        val_size = val_frac / (val_frac + test_frac)
        val_df, test_df = train_test_split(rest, test_size=(test_frac / (val_frac + test_frac)), random_state=random_state, stratify=rest['dur_bin'])
        for d in (df, train_df, val_df, test_df):
            if 'dur_bin' in d.columns:
                d.drop(columns=['dur_bin'], inplace=True)
    else:
        train_df, rest = train_test_split(df, test_size=(val_frac + test_frac), random_state=random_state)
        val_df, test_df = train_test_split(rest, test_size=(test_frac / (val_frac + test_frac)), random_state=random_state)

    outd = Path(out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(outd / "train.csv", index=False)
    val_df.to_csv(outd / "val.csv", index=False)
    test_df.to_csv(outd / "test.csv", index=False)

    def summarize(name, dff):
        total_hours = dff['duration_sec'].sum() / 3600.0 if 'duration_sec' in dff.columns else 0.0
        print(f"{name}: {len(dff)} samples, {total_hours:.2f} hrs")

    summarize("Train", train_df)
    summarize("Val", val_df)
    summarize("Test", test_df)
    return train_df, val_df, test_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    parser.add_argument("--min_dur", type=float, default=0.5)
    parser.add_argument("--max_dur", type=float, default=20.0)
    args = parser.parse_args()

    auto_split(args.manifest, args.out_dir, val_frac=args.val_frac, test_frac=args.test_frac, min_dur=args.min_dur, max_dur=args.max_dur)
