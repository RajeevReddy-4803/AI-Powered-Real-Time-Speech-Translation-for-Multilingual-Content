"""
Combine multiple processed manifest CSVs (e.g., Hindi + English) into one bilingual manifest.
Adds 'lang' column if missing and ensures no duplicate audio paths.
"""

import argparse
import pandas as pd
from pathlib import Path

def combine_manifests(manifests, out_path):
    dfs = []
    for m in manifests:
        df = pd.read_csv(m)
        lang = None
        if "lang" in df.columns:
            lang = df["lang"].iloc[0]
        else:
            if "hindi" in m.lower() or "hi" in m.lower():
                lang = "hi"
            elif "english" in m.lower() or "en" in m.lower():
                lang = "en"
            df["lang"] = lang
        print(f"Loaded {len(df)} entries from {m} ({lang})")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.drop_duplicates(subset=["wav_path"], inplace=True)
    combined.to_csv(out_path, index=False)
    print(f"\n✅ Combined manifest saved at: {out_path}")
    print(f"Total entries: {len(combined)} | Languages: {combined['lang'].unique()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple manifest CSVs into one")
    parser.add_argument("--manifests", nargs="+", required=True, help="List of manifest CSV paths")
    parser.add_argument("--out_path", required=True, help="Path to save combined manifest")
    args = parser.parse_args()

    combine_manifests(args.manifests, args.out_path)
