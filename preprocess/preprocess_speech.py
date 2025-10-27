# preprocess/preprocess_speech.py
"""
Memory-efficient preprocessing (Colab T4 friendly).
- Does: load -> optional denoise -> optional trim -> normalize -> save proc wav + manifest
- DOES NOT compute/store features (Whisper/IndicWav2Vec handle modeling features)
"""

import argparse
from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
import json
import os

import sys
from pathlib import Path

# Add project root to sys.path automatically
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from preprocess.utils_audio import (
    load_audio, save_wav, peak_normalize, trim_silence, spectral_subtract_noise_reduction
)
from preprocess.text_normalizer import normalize_by_lang

def load_transcripts_from_file(transcript_path):
    mapping = {}
    transcript_path = Path(transcript_path)
    if not transcript_path.exists():
        print(f"⚠️ Transcript not found: {transcript_path}")
        return mapping

    # CSV
    if transcript_path.suffix == ".csv":
        try:
            df = pd.read_csv(transcript_path)
            # find likely columns
            file_col = next((c for c in ["file","wav","path","audio"] if c in df.columns), None)
            text_col = next((c for c in ["transcript","text","caption"] if c in df.columns), None)
            if file_col and text_col:
                for _, r in df.iterrows():
                    mapping[Path(str(r[file_col])).stem] = str(r[text_col])
                return mapping
        except Exception:
            pass

    # generic text: pipe-delimited or tab-separated or ("id" "text")
    with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "|" in line:
                k, v = line.split("|", 1)
                mapping[k.strip()] = v.strip()
            elif "\t" in line:
                k, v = line.split("\t", 1)
                mapping[Path(k.strip()).stem] = v.strip()
            elif line.startswith("("):
                # format: ( id "text" )
                import re
                found = re.findall(r'"([^"]+)"', line)
                if len(found) >= 1:
                    # sometimes first token is id, sometimes both id and text quoted
                    parts = line.replace("(", "").replace(")", "").strip().split()
                    if len(parts) >= 1:
                        id_token = parts[0].strip('"')
                        text = found[-1]
                        mapping[id_token] = text
            else:
                parts = line.split()
                if len(parts) >= 2:
                    mapping[parts[0]] = " ".join(parts[1:])
    return mapping

def process_file(in_path: Path, out_root: Path, cfg, lang):
    try:
        rel = in_path.relative_to(cfg['input_dir'])
        out_dir = out_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        base = in_path.stem

        samples, sr = load_audio(str(in_path), sr=cfg['target_sr'], mono=True)

        # optional denoise
        if cfg.get('do_denoise', False):
            samples = spectral_subtract_noise_reduction(samples, sr=sr, prop_decrease=cfg.get('denoise_prop', 0.8))

        # normalize & optional trim
        samples = peak_normalize(samples)
        if cfg.get('do_trim', False):
            samples = trim_silence(samples, sr=sr, top_db=cfg.get('silence_top_db', 30))
            samples = peak_normalize(samples)

        if len(samples) < int(0.3 * sr):
            return {'input_path': str(in_path), 'error': 'too_short'}

        out_proc = out_dir / f"{base}_proc.wav"
        save_wav(out_proc, samples, sr=sr, subtype=cfg.get('wav_subtype', 'PCM_16'))

        # transcript
        transcript = ""
        if base in cfg.get('transcript_map', {}):
            transcript = normalize_by_lang(cfg['transcript_map'][base], lang)
        else:
            txt_src = in_path.with_suffix(".txt")
            if txt_src.exists():
                t = txt_src.read_text(encoding="utf-8", errors="ignore").strip()
                transcript = normalize_by_lang(t, lang)

        if transcript:
            (out_dir / f"{base}.txt").write_text(transcript, encoding="utf-8")

        return {
            "id": base,
            "wav_path": str(out_proc),
            "duration_sec": round(len(samples) / sr, 3),
            "transcript": transcript,
            "lang": lang
        }
    except Exception as e:
        return {'input_path': str(in_path), 'error': str(e)}

def find_audio_files(root):
    p = Path(root)
    exts = ('.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg')
    files = []
    for ext in exts:
        files.extend(list(p.rglob(f"*{ext}")))
    return files

def run_preprocessing(input_dir, output_dir, lang='en', n_jobs=2, transcript_file=None, do_denoise=False, do_trim=False, silence_db=30.0):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_map = {}
    if transcript_file:
        transcript_map = load_transcripts_from_file(transcript_file)

    cfg = {
        'input_dir': input_dir,
        'target_sr': 16000,
        'wav_subtype': 'PCM_16',
        'transcript_map': transcript_map,
        'do_denoise': do_denoise,
        'denoise_prop': 0.8,
        'do_trim': do_trim,
        'silence_top_db': silence_db
    }

    files = find_audio_files(input_dir)
    print(f"🔍 Found {len(files)} audio files under {input_dir}")

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_file)(f, output_dir, cfg, lang) for f in tqdm(files)
    )

    records = [r for r in results if r and 'error' not in r]
    errors = [r for r in results if r and 'error' in r]

    if records:
        df = pd.DataFrame.from_records(records)
        manifest_path = output_dir / "preprocess_manifest.csv"
        df.to_csv(manifest_path, index=False)
        print(f"✅ Manifest saved to {manifest_path}")

    if errors:
        print(f"⚠️ {len(errors)} errors. Sample:")
        for e in errors[:5]:
            print(e)

    return records, errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--transcript_file", type=str, default=None)
    parser.add_argument("--denoise", action="store_true", help="Enable spectral denoise (use only if recordings are noisy)")
    parser.add_argument("--trim", action="store_true", help="Enable silence trimming")
    parser.add_argument("--silence_db", type=float, default=30.0)
    args = parser.parse_args()

    run_preprocessing(args.input_dir, args.output_dir, lang=args.lang, n_jobs=args.n_jobs,
                      transcript_file=args.transcript_file, do_denoise=args.denoise, do_trim=args.trim, silence_db=args.silence_db)
