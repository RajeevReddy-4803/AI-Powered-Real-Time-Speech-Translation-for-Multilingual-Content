import re
import csv
from pathlib import Path

def map_hindi_text(input_txt, wav_dir, output_csv):
    print(f"🔍 Mapping Hindi transcripts from {input_txt}")
    entries = []
    with open(input_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            txt_id = Path(parts[0]).stem  # e.g. train_hindimale_00001
            transcript = parts[1].strip()
            wav_file = Path(wav_dir) / f"{txt_id}.wav"
            if wav_file.exists():
                entries.append((wav_file.as_posix(), transcript))
    print(f"✅ Mapped {len(entries)} Hindi transcripts.")
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "transcript"])
        writer.writerows(entries)
    print(f"Hindi mapping saved to: {output_csv}\n")

def map_english_text(input_txt, wav_dir, output_csv):
    print(f"🔍 Mapping English transcripts from {input_txt}")
    entries = []
    pattern = re.compile(r'\(\s*(\S+)\s+"(.+?)"\s*\)')
    with open(input_txt, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if not match:
                continue
            txt_id, transcript = match.groups()
            wav_file = Path(wav_dir) / f"{txt_id}.wav"
            if wav_file.exists():
                entries.append((wav_file.as_posix(), transcript.strip()))
    print(f"✅ Mapped {len(entries)} English transcripts.")
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "transcript"])
        writer.writerows(entries)
    print(f"English mapping saved to: {output_csv}\n")

if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1] / "raw_data"

    map_hindi_text(
        input_txt=base / "hindi" / "hindi_male_mono.txt",
        wav_dir=base / "hindi" / "wav",
        output_csv=base / "hindi" / "text_mapped.csv"
    )

    map_english_text(
        input_txt=base / "english" / "text.done.data",
        wav_dir=base / "english" / "wav",
        output_csv=base / "english" / "text_mapped.csv"
    )
