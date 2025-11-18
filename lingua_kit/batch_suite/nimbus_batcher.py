"""
Batch Suite — Multilingual Speech-to-Speech Translator
Automatically transcribes audio files and translates to 12+ target languages.

This edition is wired directly into the fine-tuned Whisper checkpoints stored in
`models/whisper/whisper_finetuned` while still retaining the optional Google STT
fallback for lightweight experimentation.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from tqdm import tqdm

from lingua_kit.config import (
    ASR_COMBINED_DIR,
    BASE_DIR,
    DEFAULT_STT_METHOD,
    TARGET_LANGS,
    TRANSLATION_OUTPUT_DIR,
)
from lingua_kit.substrate.audio import ensure_wav
from lingua_kit.substrate.stt import SpeechToTextEngine
from lingua_kit.substrate.translate import get_target_langs, synthesize_speech, translate_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch translate audio files to 12+ languages.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=BASE_DIR / "lingua_kit" / "batch_suite" / "data",
        help="Directory that holds source audio files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=TRANSLATION_OUTPUT_DIR / "batch_outputs",
        help="Directory that will receive synthesized audio files.",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=TRANSLATION_OUTPUT_DIR / "logs",
        help="Directory to store CSV logs.",
    )
    parser.add_argument(
        "--stt_backend",
        choices=["whisper", "google", "auto"],
        default=DEFAULT_STT_METHOD,
        help="Speech-to-text backend to use.",
    )
    parser.add_argument(
        "--target_langs",
        type=str,
        default=None,
        help="Comma separated list of language codes. Defaults to all TARGET_LANGS.",
    )
    parser.add_argument(
        "--transcripts_csv",
        type=Path,
        default=None,
        help="Optional CSV with columns [audio_path|input_file|out_wav] + [transcript|prediction].",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=45,
        help="Maximum number of audio files to process (protects long runs).",
    )
    parser.add_argument(
        "--manifest_csv",
        type=Path,
        default=ASR_COMBINED_DIR / "test_manifest.csv",
        help="Manifest with `out_wav` column. Used when input_dir is omitted.",
    )
    return parser.parse_args()


def discover_audio_files(input_dir: Path, manifest_csv: Optional[Path]) -> Iterable[Path]:
    if manifest_csv and manifest_csv.exists():
        df = pd.read_csv(manifest_csv)
        if "out_wav" in df.columns:
            for item in df["out_wav"]:
                candidate = Path(item)
                if candidate.exists():
                    yield candidate
        return

    if input_dir.exists():
        for pattern in ("*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg"):
            for path in input_dir.glob(pattern):
                yield path
    else:
        raise FileNotFoundError(
            f"Input directory '{input_dir}' does not exist and manifest_csv was not provided."
        )


def load_transcript_overrides(transcripts_csv: Optional[Path]) -> Dict[str, str]:
    if not transcripts_csv or not transcripts_csv.exists():
        return {}
    df = pd.read_csv(transcripts_csv)
    path_column = next((col for col in ["audio_path", "out_wav", "input_file", "file"] if col in df.columns), None)
    text_column = next((col for col in ["transcript", "prediction", "recognized_text"] if col in df.columns), None)
    if not text_column:
        raise ValueError(
            f"{transcripts_csv} must contain a transcript column "
            "(one of transcript/prediction/recognized_text)."
        )
    overrides: Dict[str, str] = {}
    if path_column:
        for _, row in df.iterrows():
            audio_path = Path(row[path_column])
            key = audio_path.stem
            overrides[key] = str(row[text_column])
    else:
        print(
            f"⚠️ No audio path column present in {transcripts_csv}. "
            "Transcripts will be applied in file order which may be risky."
        )
    return overrides


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.log_dir / f"translations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    target_langs = get_target_langs(args.target_langs)
    stt_engine = SpeechToTextEngine(method=args.stt_backend)
    overrides = load_transcript_overrides(args.transcripts_csv)

    audio_iter = list(discover_audio_files(args.input_dir, args.manifest_csv))
    if not audio_iter:
        print(f"❌ No audio files found in '{args.input_dir}'.")
        return

    if len(audio_iter) > args.max_files:
        print(
            f"\n⚠️ Found {len(audio_iter)} files, limiting to {args.max_files} "
            "to keep runtime manageable (~1 hour)."
        )
        audio_iter = audio_iter[: args.max_files]

    log_entries = []

    for idx, audio_path in enumerate(tqdm(audio_iter, desc="Processing files")):
        print("\n" + "=" * 60)
        print(f"🎧 Processing: {audio_path.name}")
        print("=" * 60)

        wav_path = ensure_wav(audio_path)
        if not wav_path:
            continue

        transcript = overrides.get(audio_path.stem)
        if not transcript:
            transcript = stt_engine.transcribe(wav_path)
        if not transcript:
            print("⚠️ Speech recognition failed, skipping file.")
            continue

        print(f"🗣 Recognized: {transcript}")

        translations = {}
        for code, lang_name in target_langs.items():
            translated = translate_text(transcript, code)
            if translated:
                translations[code] = translated
                out_mp3 = args.output_dir / f"{audio_path.stem}_{code}.mp3"
                synthesize_speech(translated, code, out_mp3)

        log_entries.append(
            {
                "input_file": audio_path.name,
                "recognized_text": transcript,
                **{f"translation_{code}": translations.get(code, "") for code in target_langs.keys()},
            }
        )

        if wav_path != audio_path and wav_path.exists():
            try:
                wav_path.unlink()
            except OSError:
                pass

    if log_entries:
        fieldnames = ["input_file", "recognized_text"] + [f"translation_{code}" for code in target_langs.keys()]
        with log_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_entries)
        print(f"\n📝 Translation log saved: {log_file}")

    print("\n" + "=" * 60)
    print("🎉 Batch translation complete!")
    print(f"📁 Output files saved in: {args.output_dir}")
    print(f"📊 Total files generated: {len(audio_iter) * len(target_langs)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

