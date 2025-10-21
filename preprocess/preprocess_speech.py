# preprocess/preprocess_speech.py
import argparse
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# local imports
from utils_audio import (
    load_audio, save_wav, peak_normalize, trim_silence,
    spectral_subtract_noise_reduction, compute_log_mel, compute_mfcc
)
from text_normalizer import normalize_by_lang


def load_transcripts_from_file(transcript_path):
    """Load mapping from either text_mapped.csv (file, transcript) or id|text."""
    mapping = {}
    transcript_path = Path(transcript_path)
    if not transcript_path.exists():
        print(f"⚠️ Transcript file not found at {transcript_path}")
        return mapping

    if transcript_path.suffix == ".csv":
        # Try CSV with headers
        try:
            df = pd.read_csv(transcript_path)
            if "file" in df.columns and "transcript" in df.columns:
                for _, row in df.iterrows():
                    file_path = Path(row["file"])
                    key = file_path.stem  # "train_hindimale_00001"
                    mapping[key] = str(row["transcript"])
                print(f"Loaded {len(mapping)} transcript entries from CSV: {transcript_path}")
                return mapping
        except Exception as e:
            print(f"⚠️ Failed to read CSV format: {e}")

    # Fallback to pipe-delimited
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                key, text = line.strip().split("|", 1)
                mapping[key.strip()] = text.strip()

    print(f"Loaded {len(mapping)} transcript entries from text file: {transcript_path}")
    return mapping



def process_file(in_path: Path, out_root: Path, config, lang):
    """Process a single wav file — normalize, trim, extract features, attach transcript."""
    try:
        rel = in_path.relative_to(config['input_dir'])
        out_dir = out_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        basename = in_path.stem

        out_proc_wav = out_dir / f"{basename}_proc.wav"
        out_logmel = out_dir / f"{basename}_logmel.npy"
        out_mfcc = out_dir / f"{basename}_mfcc.npy"

        # load and preprocess audio
        samples, sr = load_audio(str(in_path), sr=config['target_sr'], mono=True)

        if config['do_denoise']:
            samples = spectral_subtract_noise_reduction(samples, sr=config['target_sr'], prop_decrease=config['denoise_prop'])

        if config['peak_normalize']:
            samples = peak_normalize(samples)

        samples_trimmed = trim_silence(samples, sr=config['target_sr'], top_db=config['silence_top_db'])

        if config['peak_normalize']:
            samples_trimmed = peak_normalize(samples_trimmed)

        # save processed audio
        save_wav(out_proc_wav, samples_trimmed, sr=config['target_sr'], subtype=config['wav_subtype'])

        if len(samples_trimmed) < 400:
            print(f"⚠️ Skipping {in_path.name}: too short after trimming ({len(samples_trimmed)} samples)")
            return {'input_path': str(in_path), 'error': 'too_short'}

        log_mel = compute_log_mel(samples_trimmed, sr=config['target_sr'], n_mels=config['n_mels'], fmin=config['fmin'], fmax=config['fmax'])
        mfcc = compute_mfcc(samples_trimmed, sr=config['target_sr'], n_mfcc=config['n_mfcc'])
       

        np.save(out_logmel, log_mel)
        np.save(out_mfcc, mfcc)

        # get transcript
        normalized_text = ""
        transcript_map = config.get("transcript_map", {})
        if basename in transcript_map:
            text = transcript_map[basename]
            normalized_text = normalize_by_lang(text, lang=lang)
        else:
            transcript_src = in_path.with_suffix('.txt')
            if transcript_src.exists():
                try:
                    text = transcript_src.read_text(encoding='utf-8').strip()
                except Exception:
                    text = transcript_src.read_text(errors='ignore').strip()
                normalized_text = normalize_by_lang(text, lang=lang)

        # save normalized text alongside audio
        if normalized_text:
            transcript_dst = out_dir / f"{basename}.txt"
            transcript_dst.write_text(normalized_text, encoding='utf-8')

        duration = len(samples_trimmed) / config['target_sr']
        meta = {
            'input_path': str(in_path),
            'out_wav': str(out_proc_wav),
            'logmel_npy': str(out_logmel),
            'mfcc_npy': str(out_mfcc),
            'sr': config['target_sr'],
            'duration_sec': duration,
            'transcript': normalized_text
        }
        return meta

    except Exception as e:
        return {'input_path': str(in_path), 'error': str(e)}


def find_audio_files(root, exts=('.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg')):
    """Recursively collect audio files from directory."""
    p = Path(root)
    files = []
    for ext in exts:
        files.extend(list(p.rglob(f"*{ext}")))
    return files


def run_preprocessing(input_dir, output_dir, lang='en', n_jobs=1, do_denoise=False,
                      silence_db=30, transcript_file=None):
    """Main pipeline to process all audio and transcripts."""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load transcript mapping if given
    transcript_map = {}
    if transcript_file:
        transcript_map = load_transcripts_from_file(transcript_file)

    config = {
        'input_dir': input_dir,
        'target_sr': 16000,
        'wav_subtype': 'PCM_16',
        'n_mels': 80,
        'n_mfcc': 13,
        'fmin': 20,
        'fmax': 7600,
        'silence_top_db': silence_db,
        'peak_normalize': True,
        'do_denoise': do_denoise,
        'denoise_prop': 0.8,
        'transcript_map': transcript_map
    }

    files = find_audio_files(input_dir)
    print(f"🔍 Found {len(files)} audio files under {input_dir}")

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(f, output_dir, config, lang) for f in tqdm(files)
    )

    records = [r for r in results if 'error' not in r]
    errors = [r for r in results if 'error' in r]

    if records:
        df = pd.DataFrame.from_records(records)
        manifest_path = output_dir / "preprocess_manifest.csv"
        df.to_csv(manifest_path, index=False)
        print(f"✅ Manifest saved to {manifest_path}")

    if errors:
        print(f"⚠️ Errors in {len(errors)} files. Sample:")
        for e in errors[:5]:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with raw audio files")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed files")
    parser.add_argument("--lang", default="en", help="Language code: en or hi")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel workers")
    parser.add_argument("--denoise", action='store_true', help="Apply spectral denoising")
    parser.add_argument("--silence_db", type=float, default=30.0, help="Silence trim threshold")
    parser.add_argument("--transcript_file", type=str, default=None, help="Path to transcripts.txt (optional)")
    args = parser.parse_args()

    run_preprocessing(
        args.input_dir,
        args.output_dir,
        lang=args.lang,
        n_jobs=args.n_jobs,
        do_denoise=args.denoise,
        silence_db=args.silence_db,
        transcript_file=args.transcript_file
    )
    
    