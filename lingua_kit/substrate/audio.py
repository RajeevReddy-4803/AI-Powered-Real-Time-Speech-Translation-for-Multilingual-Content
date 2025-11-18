from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from ..config import DEFAULT_SAMPLE_RATE


def _tmp_wav_path() -> Path:
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    return Path(tmp_path)


def ensure_wav(
    audio_path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    force_resample: bool = False,
) -> Optional[Path]:
    """
    Convert any supported audio/video file into a mono WAV file that is safe for STT.
    Mirrors the resilient conversion logic from the legacy research prototype while
    removing hard-coded toolchain assumptions.
    """
    source = Path(audio_path)
    if not source.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")

    if source.suffix.lower() == ".wav" and not force_resample:
        return source

    target = _tmp_wav_path()

    # 1) MoviePy with embedded ffmpeg (best for MP4/M4A/WEBM)
    if source.suffix.lower() in {".m4a", ".mp4", ".mov", ".webm", ".avi", ".mkv", ".flv"}:
        try:
            target = _convert_with_moviepy(source, target, sample_rate)
            if target:
                return target
        except Exception:
            pass

    # 2) System ffmpeg (if present)
    try:
        target = _convert_with_ffmpeg(source, target, sample_rate)
        if target:
            return target
    except Exception:
        pass

    # 3) Librosa fallback
    try:
        target = _convert_with_librosa(source, target, sample_rate)
        if target:
            return target
    except Exception:
        pass

    # 4) Pydub (requires ffmpeg but handles many formats)
    try:
        target = _convert_with_pydub(source, target, sample_rate)
        if target:
            return target
    except Exception:
        pass

    # 5) Final MoviePy attempt for mislabeled files
    try:
        target = _convert_with_moviepy(source, target, sample_rate, force=True)
        if target:
            return target
    except Exception:
        pass

    # Last resort: return original WAV
    if source.suffix.lower() == ".wav":
        return source

    raise RuntimeError(f"Failed to convert {source} to WAV. Install ffmpeg for wider format support.")


def load_audio(path: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load audio into numpy array for Whisper inference. Resamples when needed.
    """
    waveform, sr = sf.read(str(path))
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    return waveform.astype(np.float32), sr


# --- Internal helpers ----------------------------------------------------- #

def _convert_with_moviepy(
    source: Path,
    target: Path,
    sample_rate: int,
    force: bool = False,
) -> Optional[Path]:
    try:
        try:
            from moviepy import AudioFileClip  # type: ignore
        except ImportError:
            from moviepy.editor import AudioFileClip  # type: ignore

        clip = AudioFileClip(str(source))
        clip.write_audiofile(
            str(target),
            fps=sample_rate,
            nbytes=2,
            codec="pcm_s16le",
            ffmpeg_params=["-ac", "1", "-ar", str(sample_rate), "-loglevel", "error"],
        )
        clip.close()
        if target.exists() and target.stat().st_size > 100:
            return target
    except Exception:
        if force:
            raise
    return None


def _convert_with_ffmpeg(source: Path, target: Path, sample_rate: int) -> Optional[Path]:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=2)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-f",
        "wav",
        str(target),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and target.exists():
        return target
    return None


def _convert_with_librosa(source: Path, target: Path, sample_rate: int) -> Optional[Path]:
    import librosa

    y, _ = librosa.load(str(source), sr=sample_rate, mono=True)
    sf.write(str(target), y, sample_rate)
    if target.exists():
        return target
    return None


def _convert_with_pydub(source: Path, target: Path, sample_rate: int) -> Optional[Path]:
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(source))
    audio = audio.set_channels(1).set_frame_rate(sample_rate)
    audio.export(str(target), format="wav")
    if target.exists():
        return target
    return None


