from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from lingua_kit.config import BASE_DIR
from preprocess.preprocess_speech import preprocess_dataset
from preprocess.manifest_utils import combine_manifests


@dataclass
class LanguageConfig:
    lang: str
    audio_dir: Path
    transcripts: Path
    output_dir: Path
    manifest_name: str


def run_command(cmd: list[str], label: str) -> None:
    printable = " ".join(str(p) for p in cmd)
    print(f"\n🚀 {label}\n{printable}\n")
    subprocess.run(cmd, check=True)


def orchestrate_pipeline(args: argparse.Namespace) -> None:
    languages = [
        LanguageConfig(
            lang="hi",
            audio_dir=Path(args.hi_input),
            transcripts=Path(args.hi_transcripts),
            output_dir=Path(args.hi_output),
            manifest_name=args.hi_manifest,
        ),
        LanguageConfig(
            lang="en",
            audio_dir=Path(args.en_input),
            transcripts=Path(args.en_transcripts),
            output_dir=Path(args.en_output),
            manifest_name=args.en_manifest,
        ),
    ]

    manifest_paths: Dict[str, Path] = {}

    if not args.skip_preprocess:
        print("\n🧹 Stage 1 — Preprocessing raw audio\n" + "-" * 40)
        for cfg in languages:
            manifest_paths[cfg.lang] = preprocess_dataset(
                cfg.audio_dir,
                cfg.transcripts,
                cfg.output_dir,
                cfg.lang,
                n_jobs=args.n_jobs,
                batch_size=args.batch_size,
                denoise=args.denoise,
                trim=args.trim_silence,
                manifest_name=cfg.manifest_name,
                skip_existing=args.skip_existing,
            )
    else:
        for cfg in languages:
            manifest_paths[cfg.lang] = cfg.output_dir / cfg.manifest_name

    combined_manifest = None
    if manifest_paths:
        print("\n🔗 Stage 2 — Combining manifests\n" + "-" * 40)
        combined_dir = Path(args.combined_dir)
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_manifest = combine_manifests(
            Path(manifest_paths["hi"]),
            Path(manifest_paths["en"]),
            combined_dir,
        )

    if not args.skip_train:
        run_command(
            [sys.executable, str(BASE_DIR / "models" / "whisper" / "train_whisper_asr.py")],
            "Stage 3 — Whisper fine-tuning",
        )

    if not args.skip_eval:
        run_command(
            [sys.executable, str(BASE_DIR / "models" / "whisper" / "evaluate_whisper_asr.py")],
            "Stage 4 — Model evaluation",
        )

    if not args.skip_translation and combined_manifest:
        cmd = [
            sys.executable,
            str(BASE_DIR / "lingua_kit" / "batch_suite" / "nimbus_batcher.py"),
            "--manifest_csv",
            str(combined_manifest),
            "--max_files",
            str(args.translation_files),
        ]
        if args.translation_targets:
            cmd.extend(["--target_langs", args.translation_targets])
        run_command(cmd, "Stage 5 — Sample batch translations")

    print("\n✅ Pipeline finished. Review outputs under data/translation_outputs/ and models/whisper/whisper_finetuned.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Lingua Kit pipeline end-to-end (preprocess → train → evaluate → translate)."
    )
    parser.add_argument("--n_jobs", type=int, default=2, help="Parallel workers for preprocessing")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for preprocessing chunks")
    parser.add_argument("--denoise", action="store_true", help="Apply denoising during preprocessing")
    parser.add_argument("--trim_silence", action="store_true", help="Trim leading/trailing silence during preprocessing")
    parser.add_argument("--skip_existing", action="store_true", help="Skip files whose processed WAV already exists")
    parser.add_argument("--skip_preprocess", action="store_true", help="Reuse existing manifests instead of preprocessing")
    parser.add_argument("--skip_train", action="store_true", help="Skip Whisper fine-tuning stage")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation stage")
    parser.add_argument("--skip_translation", action="store_true", help="Skip the sample translation stage")
    parser.add_argument("--translation_targets", type=str, default=None, help="Comma-separated language codes for sample translations")
    parser.add_argument("--translation_files", type=int, default=5, help="Limit for sample files translated by Batch Suite")

    parser.add_argument("--hi_input", type=str, default=str(BASE_DIR / "raw_data" / "hindi"), help="Hindi audio directory")
    parser.add_argument("--hi_transcripts", type=str, default=str(BASE_DIR / "raw_data" / "hindi" / "text_mapped.csv"), help="Hindi transcript CSV")
    parser.add_argument("--hi_output", type=str, default=str(BASE_DIR / "data" / "hindi_preprocessed"), help="Hindi preprocessing output directory")
    parser.add_argument("--hi_manifest", type=str, default="train_manifest.csv", help="Hindi manifest filename")

    parser.add_argument("--en_input", type=str, default=str(BASE_DIR / "raw_data" / "english"), help="English audio directory")
    parser.add_argument("--en_transcripts", type=str, default=str(BASE_DIR / "raw_data" / "english" / "text_mapped.csv"), help="English transcript CSV")
    parser.add_argument("--en_output", type=str, default=str(BASE_DIR / "data" / "english_preprocessed"), help="English preprocessing output directory")
    parser.add_argument("--en_manifest", type=str, default="train_manifest.csv", help="English manifest filename")

    parser.add_argument(
        "--combined_dir",
        type=str,
        default=str(BASE_DIR / "data" / "asr_combined"),
        help="Directory to store combined manifests",
    )

    return parser


if __name__ == "__main__":
    cli = build_parser()
    orchestrate_pipeline(cli.parse_args())

