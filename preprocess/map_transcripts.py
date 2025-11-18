# preprocess/map_transcripts.py
"""
Transcript-to-audio mapping for Hindi and English datasets.
Optimized for Colab (T4 GPU) and memory efficiency.
"""

import re
import csv
from pathlib import Path

def map_hindi_text(input_txt, wav_dir, output_csv):
    """
    Maps Hindi transcripts in format:
    train_hindimale_00001.txt\tमैं जीवन की राह में आगे बढ़ चला .
    """
    print(f"🔍 Mapping Hindi transcripts from: {input_txt}")
    wav_dir = Path(wav_dir)

    with open(output_txt := output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["file", "transcript"])

        count = 0
        with open(input_txt, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                txt_id = Path(parts[0]).stem.replace(".txt", "")  # remove .txt if present
                transcript = parts[1].strip()
                wav_file = wav_dir / f"{txt_id}.wav"
                if wav_file.exists():
                    writer.writerow([wav_file.as_posix(), transcript])
                    count += 1

    print(f"✅ Mapped {count} Hindi transcripts → {output_csv}\n")


def map_english_text(input_txt, wav_dir, output_csv):
    """
    Maps English transcripts in format:
    ( train_hindifullmale_00001 " There was once a merchant... " )
    """
    print(f"🔍 Mapping English transcripts from: {input_txt}")
    pattern = re.compile(r'^\(\s*(\S+)\s+"(.+?)"\s*\)$')
    wav_dir = Path(wav_dir)

    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["file", "transcript"])

        count = 0
        with open(input_txt, "r", encoding="utf-8") as fin:
            for line in fin:
                match = pattern.match(line.strip())
                if not match:
                    continue
                txt_id, transcript = match.groups()
                transcript = transcript.strip().replace("  ", " ")
                wav_file = wav_dir / f"{txt_id}.wav"
                if wav_file.exists():
                    writer.writerow([wav_file.as_posix(), transcript])
                    count += 1

    print(f"✅ Mapped {count} English transcripts → {output_csv}\n")


if __name__ == "__main__":
    # ✅ Colab dataset paths
    hindi_txt = Path("/content/drive/MyDrive/Hindi_male_mono/Hindi_male_mono/Hindi_male_mono.txt")
    hindi_wav_dir = Path("/content/drive/MyDrive/Hindi_male_mono/Hindi_male_mono/Hindi_male_audio")
    hindi_csv = Path("/content/drive/MyDrive/Hindi_male_mono/Hindi_male_mono/text_mapped.csv")

    english_txt = Path("/content/drive/MyDrive/hindi_male_english/english/text.done.data")
    english_wav_dir = Path("/content/drive/MyDrive/hindi_male_english/english/wav")
    english_csv = Path("/content/drive/MyDrive/hindi_male_english/english/text_mapped.csv")

    # ✅ Run both mappings
    map_hindi_text(hindi_txt, hindi_wav_dir, hindi_csv)
    map_english_text(english_txt, english_wav_dir, english_csv)
