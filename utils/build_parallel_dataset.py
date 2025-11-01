

import os
import glob
import pandas as pd
from pathlib import Path
import random

# ----------- config ----------
BASE = "/kaggle/working/speech_translation"
OUT_DIR = os.path.join(BASE, "data", "translation_parallel")
os.makedirs(OUT_DIR, exist_ok=True)

# Candidate CSV paths (common ones you have). Script will pick the first that exists.
candidates = {
    "hindi": [
        os.path.join(BASE, "data", "asr_outputs", "hindi_asr.csv"),
        os.path.join(BASE, "data", "processed", "hindi", "text_mapped.csv"),
        os.path.join(BASE, "raw_data", "hindi", "Hindi_male_mono", "text_mapped.csv"),
        # add others if needed
    ],
    "english": [
        os.path.join(BASE, "data", "asr_outputs", "english_asr.csv"),
        os.path.join(BASE, "data", "processed", "english", "text_mapped.csv"),
        os.path.join(BASE, "raw_data", "english", "english", "text_mapped.csv"),
    ]
}

def find_existing(path_list):
    for p in path_list:
        if os.path.exists(p):
            return p
    return None

hindi_csv = find_existing(candidates["hindi"])
english_csv = find_existing(candidates["english"])

if hindi_csv is None or english_csv is None:
    print("⚠️ Could not auto-locate both Hindi and English CSVs.")
    print("Searched candidates. Please set the correct paths manually in the script.")
    print("Hindi candidates tried:", candidates["hindi"])
    print("English candidates tried:", candidates["english"])
    raise SystemExit(1)

print("Using Hindi CSV:", hindi_csv)
print("Using English CSV:", english_csv)

# ---------- read robustly ----------
def load_csv(path):
    df = pd.read_csv(path, low_memory=False)
    # lower-case column names
    df.columns = [c.strip().lower() for c in df.columns]
    return df

dhi = load_csv(hindi_csv)
den = load_csv(english_csv)

# heuristics to pick file/text columns
def guess_file_col(df):
    for c in ["wav_path", "file", "audio_path", "filename", "path", "wav"]:
        if c in df.columns:
            return c
    return df.columns[0]

def guess_text_col(df):
    for c in ["text", "asr_text", "transcript", "transcription", "sentence"]:
        if c in df.columns:
            return c
    # fallback: any column that looks like long text
    for c in df.columns:
        if df[c].dtype == object and df[c].astype(str).str.len().mean() > 5:
            return c
    return df.columns[1] if len(df.columns) > 1 else df.columns[0]

fcol_hi = guess_file_col(dhi)
tcol_hi = guess_text_col(dhi)
fcol_en = guess_file_col(den)
tcol_en = guess_text_col(den)

print("Hindi file col:", fcol_hi, "text col:", tcol_hi)
print("English file col:", fcol_en, "text col:", tcol_en)

# compute basenames (stem, without extension)
dhi["basename"] = dhi[fcol_hi].astype(str).apply(lambda p: Path(p).stem)
den["basename"] = den[fcol_en].astype(str).apply(lambda p: Path(p).stem)

# inner join on basename
merged = pd.merge(
    dhi[[ "basename", tcol_hi ]].rename(columns={tcol_hi: "hi_text"}),
    den[[ "basename", tcol_en ]].rename(columns={tcol_en: "en_text" }),
    on="basename",
    how="inner"
).dropna().reset_index(drop=True)

print(f"Aligned pairs found by basename: {len(merged)}")

if len(merged) == 0:
    # fallback: create unpaired random pairing (keeps distribution)
    print("⚠️ No basename matches — building random paired dataset of equal size (unpaired mode).")
    min_len = min(len(dhi), len(den))
    random.seed(42)
    dhi_s = dhi.sample(n=min_len, random_state=42).reset_index(drop=True)
    den_s = den.sample(n=min_len, random_state=42).reset_index(drop=True)
    merged = pd.DataFrame({
        "basename": [f"pair_{i}" for i in range(min_len)],
        "hi_text": dhi_s[tcol_hi].astype(str),
        "en_text": den_s[tcol_en].astype(str),
    })

# minimal cleaning: strip and drop empty
merged["hi_text"] = merged["hi_text"].astype(str).str.strip()
merged["en_text"] = merged["en_text"].astype(str).str.strip()
merged = merged[(merged["hi_text"] != "") & (merged["en_text"] != "")].reset_index(drop=True)

print("After cleaning, usable pairs:", len(merged))

# shuffle and optionally limit (if you want to experiment with subset)
merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

# split
n = len(merged)
n_val = max(100, int(0.10 * n))
n_test = max(100, int(0.10 * n))
n_train = max(0, n - n_val - n_test)

train = merged.iloc[:n_train].reset_index(drop=True)
val = merged.iloc[n_train:n_train + n_val].reset_index(drop=True)
test = merged.iloc[n_train + n_val:n_train + n_val + n_test].reset_index(drop=True)

print("Split sizes -> train:", len(train), "val:", len(val), "test:", len(test))

# save CSVs (columns: src,tgt,id)
train_out = os.path.join(OUT_DIR, "train.csv")
val_out   = os.path.join(OUT_DIR, "val.csv")
test_out  = os.path.join(OUT_DIR, "test.csv")

train.rename(columns={"hi_text": "src", "en_text": "tgt"}, inplace=True)
val.rename(columns={"hi_text": "src", "en_text": "tgt"}, inplace=True)
test.rename(columns={"hi_text": "src", "en_text": "tgt"}, inplace=True)

train[["src","tgt","basename"]].to_csv(train_out, index=False)
val[["src","tgt","basename"]].to_csv(val_out, index=False)
test[["src","tgt","basename"]].to_csv(test_out, index=False)

print("✅ Saved parallel splits to:", OUT_DIR)
print("train.csv rows:", len(train))
print("val.csv rows:", len(val))
print("test.csv rows:", len(test))

