# preprocess/utils_audio.py
"""
Lightweight audio utilities for Colab T4.
- Prefer torchaudio for fast I/O; fallback to librosa/soundfile.
- Provides optional silence trimming and simple spectral denoise.
- Returns numpy.float32 arrays; avoids large in-memory objects.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

# lazy imports
_torchaudio = None
_librosa = None

TARGET_SR = 16000
EPS = 1e-8
WIN_LENGTH = 400
HOP_LENGTH = 160

def _import_torchaudio():
    global _torchaudio
    if _torchaudio is None:
        try:
            import torchaudio
            _torchaudio = torchaudio
        except Exception:
            _torchaudio = None
    return _torchaudio

def _import_librosa():
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
    return _librosa

def load_audio(path, sr=TARGET_SR, mono=True):
    """
    Load audio and return (samples: np.float32, sr).
    Prefer torchaudio; fallback to librosa.
    """
    path = str(path)
    ta = _import_torchaudio()
    if ta is not None:
        try:
            waveform, orig_sr = ta.load(path)  # [channels, samples]
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0)
            samples = waveform.cpu().numpy().astype(np.float32)
            if int(orig_sr) != int(sr):
                resampler = ta.transforms.Resample(orig_freq=int(orig_sr), new_freq=int(sr))
                waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
                samples = waveform.cpu().numpy().astype(np.float32)
            return samples, int(sr)
        except Exception:
            pass
    # librosa fallback
    lb = _import_librosa()
    y, orig_sr = lb.load(path, sr=None, mono=mono)
    if mono and y.ndim > 1:
        y = y.mean(axis=0)
    if int(orig_sr) != int(sr):
        try:
            y = lb.resample(y, orig_sr, sr)
        except TypeError:
            y = lb.core.resample(y, orig_sr, sr)
    return y.astype(np.float32), int(sr)

def save_wav(path, samples, sr=TARGET_SR, subtype="PCM_16"):
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    sf.write(str(path), samples, sr, subtype=subtype)

def peak_normalize(samples):
    s = np.asarray(samples, dtype=np.float32)
    peak = np.max(np.abs(s)) + EPS
    return (s / peak).astype(np.float32)

def trim_silence(samples, sr=TARGET_SR, top_db=30):
    """
    Trim silent segments. Uses librosa.effects.split if available, else crude energy method.
    """
    lb = _import_librosa()
    try:
        intervals = lb.effects.split(samples, top_db=top_db, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH)
        if intervals.size == 0:
            return samples
        parts = [samples[s:e] for s, e in intervals]
        return np.concatenate(parts).astype(np.float32)
    except Exception:
        # fallback: frame energy threshold
        s = np.asarray(samples)
        frame_len = WIN_LENGTH
        hop = HOP_LENGTH
        energies = []
        for i in range(0, len(s), hop):
            f = s[i:i+frame_len]
            energies.append(np.mean(f**2))
        energies = np.array(energies)
        thresh = np.percentile(energies, 10)
        keep_idxs = np.where(energies > thresh)[0]
        if keep_idxs.size == 0:
            return s
        kept = []
        for idx in keep_idxs:
            st = idx*hop
            ed = min(len(s), st+frame_len)
            kept.append(s[st:ed])
        return np.concatenate(kept).astype(np.float32)

def spectral_subtract_noise_reduction(samples, sr=TARGET_SR, prop_decrease=0.8):
    """
    Simple spectral subtraction denoise (numpy, uses librosa if available).
    Use sparingly; may distort some recordings.
    """
    lb = _import_librosa()
    try:
        S = lb.stft(samples, n_fft=1024, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        mag, phase = np.abs(S), np.angle(S)
        noise_frames = max(1, int((0.5 * sr - WIN_LENGTH) // HOP_LENGTH))
        noise_mag = np.median(mag[:, :noise_frames], axis=1, keepdims=True)
        mag_denoised = np.maximum(mag - prop_decrease * noise_mag, 0.0)
        Sdn = mag_denoised * np.exp(1j * phase)
        y = lb.istft(Sdn, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, length=len(samples))
        return y.astype(np.float32)
    except Exception:
        return samples.astype(np.float32)
