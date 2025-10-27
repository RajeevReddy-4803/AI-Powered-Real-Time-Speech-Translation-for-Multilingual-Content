# scripts/run_all_preprocessing.py
"""
Orchestrator to preprocess both Hindi & English directories and produce one bilingual manifest.
Edit the DRIVE paths below or pass them as args.
"""

import subprocess
import argparse
from pathlib import Path

def run_cmd(cmd):
    print("RUN:", " ".join(cmd))
    r = subprocess.run(cmd, shell=False)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hi_input", default="/content/drive/MyDrive/Hindi_male_mono/Hindi_male_mono", help="Hindi raw folder")
    parser.add_argument("--hi_out", default="/content/drive/MyDrive/Hindi_male_mono/Hindi_male_mono/processed", help="Hindi processed out")
    parser.add_argument("--hi_transcript", default="/content/drive/MyDrive/Hindi_male_mono/Hindi_male_mono/Hindi_male_mono.txt")
    parser.add_argument("--en_input", default="/content/drive/MyDrive/hindi_male_english/english/wav", help="English raw wavs")
    parser.add_argument("--en_out", default="/content/drive/MyDrive/hindi_male_english/english/processed")
    parser.add_argument("--en_transcript", default="/content/drive/MyDrive/hindi_male_english/english/text.done.data")
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--save_features", action='store_true')
    parser.add_argument("--denoise", action='store_true')
    parser.add_argument("--out_bilingual", default="/content/drive/MyDrive/AI-Powered-Real-Time-Speech-Translation-for-Multilingual-Content/bilingual_manifest.csv")
    args = parser.parse_args()

    # Run Hindi preprocess
    cmd_hi = [
        "python", "preprocess/preprocess_speech.py",
        "--input_dir", args.hi_input,
        "--output_dir", args.hi_out,
        "--lang", "hi",
        "--n_jobs", str(args.n_jobs),
        "--transcript_file", args.hi_transcript
    ]
    if args.denoise:
        cmd_hi.append("--denoise")
    if args.save_features:
        cmd_hi.append("--save_features")

    # Run English preprocess
    cmd_en = [
        "python", "preprocess/preprocess_speech.py",
        "--input_dir", args.en_input,
        "--output_dir", args.en_out,
        "--lang", "en",
        "--n_jobs", str(args.n_jobs),
        "--transcript_file", args.en_transcript
    ]
    if args.denoise:
        cmd_en.append("--denoise")
    if args.save_features:
        cmd_en.append("--save_features")

    # execute
    run_cmd(cmd_hi)
    run_cmd(cmd_en)

    # merge
    merge_cmd = [
        "python", "scripts/rebuild_and_merge_manifests.py",
        "--manifests", str(Path(args.hi_out) / "preprocess_manifest.csv"), str(Path(args.en_out) / "preprocess_manifest.csv"),
        "--out", args.out_bilingual
    ]
    run_cmd(merge_cmd)
    print("All preprocessing done. Bilingual manifest at:", args.out_bilingual)
