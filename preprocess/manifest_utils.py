# preprocess/manifest_utils.py
"""
Enhanced manifest splitter for ASR/TTS datasets.
Performs train/val/test split with duration filtering,
language support, and summary statistics.
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch  # optional if you're using PyTorch later

random.seed(42)
np.random.seed(42)
try:
    torch.manual_seed(42)
except Exception:
    pass


def summarize_manifest(df, name):
    """Print dataset summary"""
    total_dur = df["duration_sec"].sum() / 3600.0 if "duration_sec" in df.columns else 0
    print(f"\n📊 {name} set: {len(df)} samples | {total_dur:.2f} hrs total | "
          f"avg {df['duration_sec'].mean():.2f}s per clip")

def split_manifest(
    manifest_csv,
    out_dir,
    val_frac=0.05,
    test_frac=0.05,
    random_state=42,
    min_dur=0.5,
    max_dur=20.0,
    add_lang_col=None
):
    """Split manifest into train/val/test with filtering & stats"""
    df = pd.read_csv(manifest_csv)
    print(f"Loaded {len(df)} entries from {manifest_csv}")

    # --- duration filter ---
    if "duration_sec" in df.columns:
        before = len(df)
        df = df[(df["duration_sec"] >= min_dur) & (df["duration_sec"] <= max_dur)]
        print(f"Filtered durations outside [{min_dur}s, {max_dur}s]: {before} → {len(df)}")

    # --- add language column if requested ---
    if add_lang_col and "lang" not in df.columns:
        df["lang"] = add_lang_col
        print(f"Added language column: {add_lang_col}")

    # --- stratified split by duration bins ---
    if "duration_sec" in df.columns:
        df["dur_bin"] = pd.qcut(df["duration_sec"].clip(lower=min_dur), q=5, duplicates="drop")
        train, rest = train_test_split(
            df,
            test_size=(val_frac + test_frac),
            random_state=random_state,
            stratify=df["dur_bin"]
        )
        val_size = val_frac / (val_frac + test_frac)
        val, test = train_test_split(
            rest,
            test_size=(test_frac / (val_frac + test_frac)),
            random_state=random_state,
            stratify=rest["dur_bin"]
        )
        for d in (df, train, val, test):
            if "dur_bin" in d.columns:
                d.drop(columns=["dur_bin"], inplace=True)
    else:
        train, rest = train_test_split(df, test_size=(val_frac + test_frac), random_state=random_state)
        val, test = train_test_split(rest, test_size=(test_frac / (val_frac + test_frac)), random_state=random_state)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- save splits ---
    train.to_csv(out_dir / "train_manifest.csv", index=False)
    val.to_csv(out_dir / "val_manifest.csv", index=False)
    test.to_csv(out_dir / "test_manifest.csv", index=False)

    print(f"\n✅ Saved train/val/test manifests to {out_dir}")
    summarize_manifest(train, "Train")
    summarize_manifest(val, "Validation")
    summarize_manifest(test, "Test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and clean manifest CSV for ASR/TTS")
    parser.add_argument("--manifest", required=True, help="Path to preprocess_manifest.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory for split manifests")
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    parser.add_argument("--min_dur", type=float, default=0.5)
    parser.add_argument("--max_dur", type=float, default=20.0)
    parser.add_argument("--lang", type=str, default=None, help="Force language column (en|hi)")
    args = parser.parse_args()

    split_manifest(
        args.manifest,
        args.out_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        add_lang_col=args.lang
    )
# preprocess/manifest_utils.py
"""
Enhanced manifest splitter for ASR/TTS datasets.
Performs train/val/test split with duration filtering,
language support, and summary statistics.
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch  # optional if you're using PyTorch later

random.seed(42)
np.random.seed(42)
try:
    torch.manual_seed(42)
except Exception:
    pass


def summarize_manifest(df, name):
    """Print dataset summary"""
    total_dur = df["duration_sec"].sum() / 3600.0 if "duration_sec" in df.columns else 0
    print(f"\n📊 {name} set: {len(df)} samples | {total_dur:.2f} hrs total | "
          f"avg {df['duration_sec'].mean():.2f}s per clip")

def split_manifest(
    manifest_csv,
    out_dir,
    val_frac=0.05,
    test_frac=0.05,
    random_state=42,
    min_dur=0.5,
    max_dur=20.0,
    add_lang_col=None
):
    """Split manifest into train/val/test with filtering & stats"""
    df = pd.read_csv(manifest_csv)
    print(f"Loaded {len(df)} entries from {manifest_csv}")

    # --- duration filter ---
    if "duration_sec" in df.columns:
        before = len(df)
        df = df[(df["duration_sec"] >= min_dur) & (df["duration_sec"] <= max_dur)]
        print(f"Filtered durations outside [{min_dur}s, {max_dur}s]: {before} → {len(df)}")

    # --- add language column if requested ---
    if add_lang_col and "lang" not in df.columns:
        df["lang"] = add_lang_col
        print(f"Added language column: {add_lang_col}")

    # --- stratified split by duration bins ---
    if "duration_sec" in df.columns:
        df["dur_bin"] = pd.qcut(df["duration_sec"].clip(lower=min_dur), q=5, duplicates="drop")
        train, rest = train_test_split(
            df,
            test_size=(val_frac + test_frac),
            random_state=random_state,
            stratify=df["dur_bin"]
        )
        val_size = val_frac / (val_frac + test_frac)
        val, test = train_test_split(
            rest,
            test_size=(test_frac / (val_frac + test_frac)),
            random_state=random_state,
            stratify=rest["dur_bin"]
        )
        for d in (df, train, val, test):
            if "dur_bin" in d.columns:
                d.drop(columns=["dur_bin"], inplace=True)
    else:
        train, rest = train_test_split(df, test_size=(val_frac + test_frac), random_state=random_state)
        val, test = train_test_split(rest, test_size=(test_frac / (val_frac + test_frac)), random_state=random_state)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- save splits ---
    train.to_csv(out_dir / "train_manifest.csv", index=False)
    val.to_csv(out_dir / "val_manifest.csv", index=False)
    test.to_csv(out_dir / "test_manifest.csv", index=False)

    print(f"\n✅ Saved train/val/test manifests to {out_dir}")
    summarize_manifest(train, "Train")
    summarize_manifest(val, "Validation")
    summarize_manifest(test, "Test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and clean manifest CSV for ASR/TTS")
    parser.add_argument("--manifest", required=True, help="Path to preprocess_manifest.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory for split manifests")
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    parser.add_argument("--min_dur", type=float, default=0.5)
    parser.add_argument("--max_dur", type=float, default=20.0)
    parser.add_argument("--lang", type=str, default=None, help="Force language column (en|hi)")
    args = parser.parse_args()

    split_manifest(
        args.manifest,
        args.out_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        add_lang_col=args.lang
    )
# preprocess/manifest_utils.py
"""
Enhanced manifest splitter for ASR/TTS datasets.
Performs train/val/test split with duration filtering,
language support, and summary statistics.
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch  # optional if you're using PyTorch later

random.seed(42)
np.random.seed(42)
try:
    torch.manual_seed(42)
except Exception:
    pass


def summarize_manifest(df, name):
    """Print dataset summary"""
    total_dur = df["duration_sec"].sum() / 3600.0 if "duration_sec" in df.columns else 0
    print(f"\n📊 {name} set: {len(df)} samples | {total_dur:.2f} hrs total | "
          f"avg {df['duration_sec'].mean():.2f}s per clip")

def split_manifest(
    manifest_csv,
    out_dir,
    val_frac=0.05,
    test_frac=0.05,
    random_state=42,
    min_dur=0.5,
    max_dur=20.0,
    add_lang_col=None
):
    """Split manifest into train/val/test with filtering & stats"""
    df = pd.read_csv(manifest_csv)
    print(f"Loaded {len(df)} entries from {manifest_csv}")

    # --- duration filter ---
    if "duration_sec" in df.columns:
        before = len(df)
        df = df[(df["duration_sec"] >= min_dur) & (df["duration_sec"] <= max_dur)]
        print(f"Filtered durations outside [{min_dur}s, {max_dur}s]: {before} → {len(df)}")

    # --- add language column if requested ---
    if add_lang_col and "lang" not in df.columns:
        df["lang"] = add_lang_col
        print(f"Added language column: {add_lang_col}")

    # --- stratified split by duration bins ---
    if "duration_sec" in df.columns:
        df["dur_bin"] = pd.qcut(df["duration_sec"].clip(lower=min_dur), q=5, duplicates="drop")
        train, rest = train_test_split(
            df,
            test_size=(val_frac + test_frac),
            random_state=random_state,
            stratify=df["dur_bin"]
        )
        val_size = val_frac / (val_frac + test_frac)
        val, test = train_test_split(
            rest,
            test_size=(test_frac / (val_frac + test_frac)),
            random_state=random_state,
            stratify=rest["dur_bin"]
        )
        for d in (df, train, val, test):
            if "dur_bin" in d.columns:
                d.drop(columns=["dur_bin"], inplace=True)
    else:
        train, rest = train_test_split(df, test_size=(val_frac + test_frac), random_state=random_state)
        val, test = train_test_split(rest, test_size=(test_frac / (val_frac + test_frac)), random_state=random_state)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- save splits ---
    train.to_csv(out_dir / "train_manifest.csv", index=False)
    val.to_csv(out_dir / "val_manifest.csv", index=False)
    test.to_csv(out_dir / "test_manifest.csv", index=False)

    print(f"\n✅ Saved train/val/test manifests to {out_dir}")
    summarize_manifest(train, "Train")
    summarize_manifest(val, "Validation")
    summarize_manifest(test, "Test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and clean manifest CSV for ASR/TTS")
    parser.add_argument("--manifest", required=True, help="Path to preprocess_manifest.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory for split manifests")
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    parser.add_argument("--min_dur", type=float, default=0.5)
    parser.add_argument("--max_dur", type=float, default=20.0)
    parser.add_argument("--lang", type=str, default=None, help="Force language column (en|hi)")
    args = parser.parse_args()

    split_manifest(
        args.manifest,
        args.out_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        add_lang_col=args.lang
    )
# preprocess/manifest_utils.py
"""
Enhanced manifest splitter for ASR/TTS datasets.
Performs train/val/test split with duration filtering,
language support, and summary statistics.
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch  # optional if you're using PyTorch later

random.seed(42)
np.random.seed(42)
try:
    torch.manual_seed(42)
except Exception:
    pass


def summarize_manifest(df, name):
    """Print dataset summary"""
    total_dur = df["duration_sec"].sum() / 3600.0 if "duration_sec" in df.columns else 0
    print(f"\n📊 {name} set: {len(df)} samples | {total_dur:.2f} hrs total | "
          f"avg {df['duration_sec'].mean():.2f}s per clip")

def split_manifest(
    manifest_csv,
    out_dir,
    val_frac=0.05,
    test_frac=0.05,
    random_state=42,
    min_dur=0.5,
    max_dur=20.0,
    add_lang_col=None
):
    """Split manifest into train/val/test with filtering & stats"""
    df = pd.read_csv(manifest_csv)
    print(f"Loaded {len(df)} entries from {manifest_csv}")

    # --- duration filter ---
    if "duration_sec" in df.columns:
        before = len(df)
        df = df[(df["duration_sec"] >= min_dur) & (df["duration_sec"] <= max_dur)]
        print(f"Filtered durations outside [{min_dur}s, {max_dur}s]: {before} → {len(df)}")

    # --- add language column if requested ---
    if add_lang_col and "lang" not in df.columns:
        df["lang"] = add_lang_col
        print(f"Added language column: {add_lang_col}")

    # --- stratified split by duration bins ---
    if "duration_sec" in df.columns:
        df["dur_bin"] = pd.qcut(df["duration_sec"].clip(lower=min_dur), q=5, duplicates="drop")
        train, rest = train_test_split(
            df,
            test_size=(val_frac + test_frac),
            random_state=random_state,
            stratify=df["dur_bin"]
        )
        val_size = val_frac / (val_frac + test_frac)
        val, test = train_test_split(
            rest,
            test_size=(test_frac / (val_frac + test_frac)),
            random_state=random_state,
            stratify=rest["dur_bin"]
        )
        for d in (df, train, val, test):
            if "dur_bin" in d.columns:
                d.drop(columns=["dur_bin"], inplace=True)
    else:
        train, rest = train_test_split(df, test_size=(val_frac + test_frac), random_state=random_state)
        val, test = train_test_split(rest, test_size=(test_frac / (val_frac + test_frac)), random_state=random_state)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- save splits ---
    train.to_csv(out_dir / "train_manifest.csv", index=False)
    val.to_csv(out_dir / "val_manifest.csv", index=False)
    test.to_csv(out_dir / "test_manifest.csv", index=False)

    print(f"\n✅ Saved train/val/test manifests to {out_dir}")
    summarize_manifest(train, "Train")
    summarize_manifest(val, "Validation")
    summarize_manifest(test, "Test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and clean manifest CSV for ASR/TTS")
    parser.add_argument("--manifest", required=True, help="Path to preprocess_manifest.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory for split manifests")
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    parser.add_argument("--min_dur", type=float, default=0.5)
    parser.add_argument("--max_dur", type=float, default=20.0)
    parser.add_argument("--lang", type=str, default=None, help="Force language column (en|hi)")
    args = parser.parse_args()

    split_manifest(
        args.manifest,
        args.out_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        add_lang_col=args.lang
    )
# preprocess/manifest_utils.py
"""
Enhanced manifest splitter for ASR/TTS datasets.
Performs train/val/test split with duration filtering,
language support, and summary statistics.
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch  # optional if you're using PyTorch later

random.seed(42)
np.random.seed(42)
try:
    torch.manual_seed(42)
except Exception:
    pass


def summarize_manifest(df, name):
    """Print dataset summary"""
    total_dur = df["duration_sec"].sum() / 3600.0 if "duration_sec" in df.columns else 0
    print(f"\n📊 {name} set: {len(df)} samples | {total_dur:.2f} hrs total | "
          f"avg {df['duration_sec'].mean():.2f}s per clip")

def split_manifest(
    manifest_csv,
    out_dir,
    val_frac=0.05,
    test_frac=0.05,
    random_state=42,
    min_dur=0.5,
    max_dur=20.0,
    add_lang_col=None
):
    """Split manifest into train/val/test with filtering & stats"""
    df = pd.read_csv(manifest_csv)
    print(f"Loaded {len(df)} entries from {manifest_csv}")

    # --- duration filter ---
    if "duration_sec" in df.columns:
        before = len(df)
        df = df[(df["duration_sec"] >= min_dur) & (df["duration_sec"] <= max_dur)]
        print(f"Filtered durations outside [{min_dur}s, {max_dur}s]: {before} → {len(df)}")

    # --- add language column if requested ---
    if add_lang_col and "lang" not in df.columns:
        df["lang"] = add_lang_col
        print(f"Added language column: {add_lang_col}")

    # --- stratified split by duration bins ---
    if "duration_sec" in df.columns:
        df["dur_bin"] = pd.qcut(df["duration_sec"].clip(lower=min_dur), q=5, duplicates="drop")
        train, rest = train_test_split(
            df,
            test_size=(val_frac + test_frac),
            random_state=random_state,
            stratify=df["dur_bin"]
        )
        val_size = val_frac / (val_frac + test_frac)
        val, test = train_test_split(
            rest,
            test_size=(test_frac / (val_frac + test_frac)),
            random_state=random_state,
            stratify=rest["dur_bin"]
        )
        for d in (df, train, val, test):
            if "dur_bin" in d.columns:
                d.drop(columns=["dur_bin"], inplace=True)
    else:
        train, rest = train_test_split(df, test_size=(val_frac + test_frac), random_state=random_state)
        val, test = train_test_split(rest, test_size=(test_frac / (val_frac + test_frac)), random_state=random_state)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- save splits ---
    train.to_csv(out_dir / "train_manifest.csv", index=False)
    val.to_csv(out_dir / "val_manifest.csv", index=False)
    test.to_csv(out_dir / "test_manifest.csv", index=False)

    print(f"\n✅ Saved train/val/test manifests to {out_dir}")
    summarize_manifest(train, "Train")
    summarize_manifest(val, "Validation")
    summarize_manifest(test, "Test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and clean manifest CSV for ASR/TTS")
    parser.add_argument("--manifest", required=True, help="Path to preprocess_manifest.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory for split manifests")
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    parser.add_argument("--min_dur", type=float, default=0.5)
    parser.add_argument("--max_dur", type=float, default=20.0)
    parser.add_argument("--lang", type=str, default=None, help="Force language column (en|hi)")
    args = parser.parse_args()

    split_manifest(
        args.manifest,
        args.out_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        add_lang_col=args.lang
    )
