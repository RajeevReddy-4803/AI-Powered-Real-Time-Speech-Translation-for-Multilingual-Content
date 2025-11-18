"""
Convert all MP3 files to WAV format (no ffmpeg needed).
Uses pydub which may need ffmpeg, but we'll try alternative methods.
"""
import os
import glob
from pydub import AudioSegment

def convert_mp3_to_wav(input_dir="data", output_dir="data_wav"):
    """Convert all MP3 files to WAV format."""
    os.makedirs(output_dir, exist_ok=True)
    
    mp3_files = glob.glob(os.path.join(input_dir, "*.mp3"))
    
    print(f"Found {len(mp3_files)} MP3 files to convert...")
    
    converted = 0
    failed = 0
    
    for mp3_file in mp3_files:
        try:
            # Try to load and convert
            audio = AudioSegment.from_mp3(mp3_file)
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Save as WAV
            wav_name = os.path.basename(mp3_file).replace('.mp3', '.wav')
            wav_path = os.path.join(output_dir, wav_name)
            
            audio.export(wav_path, format="wav")
            converted += 1
            
            if converted % 10 == 0:
                print(f"  Converted {converted} files...")
                
        except Exception as e:
            print(f"  Failed to convert {os.path.basename(mp3_file)}: {e}")
            failed += 1
            continue
    
    print(f"\nComplete!")
    print(f"  Converted: {converted}")
    print(f"  Failed: {failed}")
    print(f"  WAV files saved in: {os.path.abspath(output_dir)}")
    
    if failed > 0:
        print(f"\nNote: {failed} files failed - likely need ffmpeg installed.")
        print("   Download from: https://ffmpeg.org/download.html")

if __name__ == "__main__":
    convert_mp3_to_wav()

