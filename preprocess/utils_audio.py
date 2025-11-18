# preprocess/utils_audio.py
import torchaudio
import torch
import soundfile as sf
import os
import gc
from pathlib import Path

# Default parameters for ASR models (Whisper & IndicWav2Vec)
TARGET_SR = 16000
MAX_DURATION_S = 20.0  # skip overly long clips
MIN_DURATION_S = 0.5   # skip too short clips


def save_audio(path, audio, sr=16000):
    """Save waveform to disk in WAV format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio.unsqueeze(0) if audio.ndim == 1 else audio, sr)

def load_audio(path, target_sr=16000):
    """Robustly load audio using torchaudio and soundfile, no TorchCodec required."""
    path = Path(path)
    if not path.exists():
        print(f"[Error] File not found: {path}")
        return None, None

    wav, sr = None, None

    # Try torchaudio first
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception as e1:
        # Fallback to soundfile
        try:
            data, sr = sf.read(str(path))
            if data.ndim == 1:
                wav = torch.tensor(data).unsqueeze(0)
            else:
                wav = torch.tensor(data).T
        except Exception as e2:
            print(f"[Warning] Failed to load {path.name}: {e1} | {e2}")
            return None, None

    # Check validity
    if wav is None or sr is None:
        print(f"[Warning] Audio load returned None for {path}")
        return None, None

    # Resample if necessary
    if target_sr and sr != target_sr:
        try:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr
        except Exception as e:
            print(f"[Warning] Resample failed for {path.name}: {e}")
            return None, None

    # Collapse to mono tensor for downstream processing
    if wav.ndim == 2:
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)

    return wav, sr

def get_duration(path):
    """Efficiently get audio duration without loading the full file."""
    try:
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except Exception as e:
        print(f"[Warning] Could not read duration for {path}: {e}")
        return 0.0


def is_valid_audio(path):
    """Check if an audio file is valid and within duration limits."""
    dur = get_duration(path)
    return MIN_DURATION_S <= dur <= MAX_DURATION_S


def trim_silence(audio, sr=TARGET_SR, top_db=25):
    """
    Trim leading and trailing silence using torchaudio’s VAD transform.
    Fallback to energy threshold if VAD fails.
    """
    try:
        audio = audio / (audio.abs().max() + 1e-8)
        vad = torchaudio.transforms.Vad(sample_rate=sr)
        trimmed = vad(audio.unsqueeze(0)).squeeze(0)
        return trimmed
    except Exception:
        energy = audio ** 2
        mask = energy > (energy.mean() * 0.01)
        if mask.any():
            start = mask.nonzero()[0]
            end = mask.nonzero()[-1]
            return audio[start:end]
        return audio


def denoise_audio(audio, strength=0.005):
    """
    Lightweight spectral subtraction denoising.
    Fast and memory efficient for Colab.
    """
    noise_est = torch.mean(audio[audio.abs() < strength]) if (audio.abs() < strength).any() else 0.0
    denoised = audio - noise_est
    return torch.clamp(denoised, -1.0, 1.0)


def preprocess_audio(path, out_dir=None, target_sr: int = TARGET_SR):
    """Load, normalize, resample, and optionally save audio."""
    if not is_valid_audio(path):
        return None, 0.0

    waveform, sr = load_audio(path, target_sr=target_sr)
    if waveform is None:
        return None, 0.0

    duration = waveform.shape[0] / target_sr

    if out_dir:
        out_path = Path(out_dir) / Path(path).name
        save_audio(out_path, waveform, target_sr)
        del waveform
        gc.collect()
        return str(out_path), duration

    del waveform
    gc.collect()
    return path, duration


def batch_preprocess(audio_paths, out_dir=None, target_sr: int = TARGET_SR):
    """Memory-efficient generator for batch preprocessing."""
    for path in audio_paths:
        result = preprocess_audio(path, out_dir=out_dir, target_sr=target_sr)
        if result and result[0] is not None:
            yield result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio utilities for ASR preprocessing")
    parser.add_argument("--input_dir", required=True, help="Input directory containing .wav files")
    parser.add_argument("--output_dir", help="Optional output directory for processed audio")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else None

    audio_files = list(input_dir.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files in {input_dir}")

    count = 0
    for out_path, dur in batch_preprocess(audio_files, out_dir=out_dir):
        count += 1
        if count % 50 == 0:
            print(f"Processed {count} files...")

    print(f"✅ Finished preprocessing {count} files.")
