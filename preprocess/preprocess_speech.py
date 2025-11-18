"""
Preprocess speech data for ASR training.
- Supports Hindi and English
- Optional denoising and silence trimming
- Multiprocessing (n_jobs)
- Memory-efficient batch processing
"""

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from utils_audio import (
    load_audio,
    save_audio,
    trim_silence,
    denoise_audio,
    get_duration,
)
from text_normalizer import normalize_en, normalize_hi


def normalize_text(text: str, lang: str) -> str:
    """Wrapper for language-specific normalization."""
    return normalize_hi(text) if lang == "hi" else normalize_en(text)


def detect_columns(df):
    """Auto-detect path and text columns for different dataset formats."""
    lower_cols = [c.lower() for c in df.columns]
    path_col = next((c for c in df.columns if any(k in c.lower() for k in ["path", "file", "audio", "wav"])), None)
    text_col = next((c for c in df.columns if any(k in c.lower() for k in ["text", "transcript", "sentence"])), None)

    if not path_col or not text_col:
        raise ValueError(f"❌ Could not detect 'path' and 'text' columns. Found columns: {list(df.columns)}")
    return path_col, text_col


def process_row(
    row,
    path_col,
    text_col,
    output_dir,
    lang,
    base_audio_dir,
    denoise=False,
    trim=False,
    skip_existing=False,
):
    """Process a single (wav, transcript) pair."""
    wav_path = Path(str(row[path_col]))
    if not wav_path.is_absolute():
        wav_path = base_audio_dir / wav_path

    transcript = normalize_text(str(row[text_col]), lang)

    try:
        if not wav_path.exists():
            print(f"⚠️ Missing file: {wav_path}")
            return None

        out_wav = output_dir / "wavs" / wav_path.name

        if skip_existing and out_wav.exists():
            return {
                "audio_filepath": str(out_wav.resolve()),
                "text": transcript,
                "lang": lang,
                "duration": get_duration(str(out_wav)),
            }

        audio, sr = load_audio(wav_path)
        if audio is None or sr is None:
            return None

        if trim:
            audio = trim_silence(audio, sr)
        if denoise:
            audio = denoise_audio(audio)

        save_audio(out_wav, audio, sr)
        duration = audio.shape[-1] / sr if sr else 0.0

        return {
            "audio_filepath": str(out_wav.resolve()),
            "text": transcript,
            "lang": lang,
            "duration": float(duration),
        }
    except Exception as e:
        print(f"⚠️ Error processing {wav_path.name}: {e}")
        return None


def preprocess_dataset(
    input_dir,
    transcript_file,
    output_dir,
    lang,
    n_jobs=1,
    batch_size=16,
    denoise=False,
    trim=False,
    manifest_name="train_manifest.csv",
    skip_existing=False,
):
    """Main preprocessing function for ASR datasets."""
    output_dir = Path(output_dir)
    base_audio_dir = Path(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wavs").mkdir(exist_ok=True)

    df = pd.read_csv(transcript_file)
    print(f"📄 Loaded {len(df)} transcripts for {lang.upper()}")

    path_col, text_col = detect_columns(df)
    print(f"✅ Using columns → path: '{path_col}', text: '{text_col}'")

    manifest = []
    total = len(df)
    records = df.to_dict("records")

    for start in range(0, total, batch_size):
        batch = records[start:start + batch_size]
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(
                    process_row,
                    row,
                    path_col,
                    text_col,
                    output_dir,
                    lang,
                    base_audio_dir,
                    denoise,
                    trim,
                    skip_existing,
                )
                for row in batch
            ]
            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Batch {start // batch_size + 1}",
            ):
                res = f.result()
                if res:
                    manifest.append(res)

    manifest_path = output_dir / manifest_name
    if manifest:
        pd.DataFrame(manifest).to_csv(manifest_path, index=False)
        print(f"✅ Saved manifest to {manifest_path}")
    else:
        print("⚠️ No processed files — manifest not written.")

    stats = {
        "language": lang,
        "records_total": len(df),
        "records_processed": len(manifest),
        "output_dir": str(output_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "skip_existing": skip_existing,
    }

    stats_path = output_dir / "preprocess_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"📊 Total processed: {len(manifest)} / {len(df)}")
    print(f"🗂 Stats written to {stats_path}")

    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ASR speech data (Hindi/English)")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input WAV directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save preprocessed files")
    parser.add_argument("--lang", type=str, required=True, choices=["hi", "en"], help="Language code")
    parser.add_argument("--transcript_file", type=str, required=True, help="CSV file with transcript mappings")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per processing chunk")
    parser.add_argument("--denoise", action="store_true", help="Apply light denoising filter")
    parser.add_argument("--trim_silence", action="store_true", help="Trim leading and trailing silence")
    parser.add_argument("--manifest_name", type=str, default="train_manifest.csv", help="Manifest filename to write")
    parser.add_argument("--skip_existing", action="store_true", help="Skip re-processing files with cached outputs")

    args = parser.parse_args()

    preprocess_dataset(
        args.input_dir,
        args.transcript_file,
        args.output_dir,
        args.lang,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        denoise=args.denoise,
        trim=args.trim_silence,
        manifest_name=args.manifest_name,
        skip_existing=args.skip_existing,
    )
