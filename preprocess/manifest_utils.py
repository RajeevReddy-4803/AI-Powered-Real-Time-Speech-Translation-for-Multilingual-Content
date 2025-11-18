"""
Manifest creation utilities for ASR datasets (Hindi + English).
Builds CSV manifests compatible with Whisper & IndicWav2Vec training.
Optimized for memory efficiency and Google Colab usage.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from text_normalizer import normalize_by_lang      # ✅ fixed absolute import
from utils_audio import get_duration as get_audio_duration  # ✅ match your utils_audio.py function name


__all__ = ["create_manifest", "combine_manifests"]


# =============================
# 🧩 Manifest Creator
# =============================
def create_manifest(csv_path: Path, output_dir: Path, lang: str):
    """
    Create ASR manifest (CSV) from mapped transcript CSV.
    
    Parameters
    ----------
    csv_path : Path
        Input CSV with columns ['wav_path', 'transcript'].
    output_dir : Path
        Output directory for manifest files.
    lang : str
        Language code ('hi' or 'en').
    """
    df = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    print(f"📘 Creating manifest for {lang.upper()} dataset...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {lang}"):
        wav_path = Path(row["wav_path"])
        if not wav_path.exists():
            continue

        transcript = normalize_by_lang(str(row["transcript"]), lang)
        duration = get_audio_duration(str(wav_path))

        if duration is None or duration < 0.2:
            continue  # skip invalid or very short clips

        manifest.append({
            "audio_filepath": str(wav_path.resolve()),
            "duration": duration,
            "text": transcript,
            "lang": lang
        })

    manifest_df = pd.DataFrame(manifest)
    manifest_path = output_dir / f"manifest_{lang}.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"✅ Manifest saved: {manifest_path} ({len(manifest_df)} samples)")
    return manifest_path


# =============================
# 🔗 Combine Manifests
# =============================
def combine_manifests(hindi_manifest: Path, english_manifest: Path, output_dir: Path):
    """
    Combine Hindi and English manifests into a unified CSV
    for multilingual ASR training.
    """
    print("🔗 Combining Hindi and English manifests...")
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for path in [hindi_manifest, english_manifest]:
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"  → Loaded {path.name} ({len(df)} samples)")
        else:
            print(f"⚠️ Skipped missing manifest: {path}")

    if not dfs:
        print("❌ No manifests found to combine.")
        return None

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    combined_path = output_dir / "train_manifest.csv"
    combined_df.to_csv(combined_path, index=False)

    print(f"✅ Combined manifest saved: {combined_path} ({len(combined_df)} total samples)")
    return combined_path


# =============================
# 🧪 Quick Test
# =============================
if __name__ == "__main__":
    base = Path("/content/drive/MyDrive")
    hindi_csv = base / "Hindi_male_mono/Hindi_male_mono/text_mapped.csv"
    english_csv = base / "hindi_male_english/english/text_mapped.csv"
    output_dir = base / "asr_manifests"

    hi_manifest = create_manifest(hindi_csv, output_dir, "hi")
    en_manifest = create_manifest(english_csv, output_dir, "en")
    combine_manifests(hi_manifest, en_manifest, output_dir)
