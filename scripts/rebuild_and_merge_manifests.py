# scripts/rebuild_and_merge_manifests.py
import argparse
import pandas as pd
from pathlib import Path

def read_manifest(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(p)
    # normalize column names
    if 'wav_path' not in df.columns:
        for c in ('out_wav', 'wav', 'audio', 'path'):
            if c in df.columns:
                df = df.rename(columns={c: 'wav_path'})
                break
    if 'transcript' not in df.columns:
        for c in ('text', 'caption'):
            if c in df.columns:
                df = df.rename(columns={c: 'transcript'})
                break
    if 'id' not in df.columns:
        if 'file' in df.columns:
            df['id'] = df['file'].apply(lambda x: Path(str(x)).stem)
        else:
            df['id'] = df.index.astype(str)
    # ensure required columns
    for col in ['id','wav_path','duration_sec','transcript','lang']:
        if col not in df.columns:
            df[col] = "" if col == 'transcript' or col == 'lang' else None
    return df[['id','wav_path','duration_sec','transcript','lang']]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True, help="List of preproc manifest CSVs to merge")
    parser.add_argument("--out", required=True, help="Output bilingual manifest CSV")
    args = parser.parse_args()

    dfs = []
    for m in args.manifests:
        df = read_manifest(m)
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {m}")
    merged = pd.concat(dfs, ignore_index=True)
    # drop duplicates by id or wav_path
    merged.drop_duplicates(subset=['wav_path'], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged.to_csv(args.out, index=False)
    print(f"Saved merged bilingual manifest to {args.out} ({len(merged)} rows)")
