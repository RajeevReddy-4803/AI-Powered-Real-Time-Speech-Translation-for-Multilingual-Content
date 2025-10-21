# preprocess/utils_audio.py
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000
TARGET_SUBTYPE = "PCM_16"
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 80
N_MFCC = 13
EPS = 1e-8

def load_audio(path, sr=TARGET_SR, mono=True):
    samples, orig_sr = librosa.load(path, sr=None, mono=mono)
    if orig_sr != sr:
        # librosa>=0.10 uses keyword-only args for resample
        samples = librosa.resample(y=samples, orig_sr=orig_sr, target_sr=sr)
    return samples, sr

def save_wav(path, samples, sr=TARGET_SR, subtype=TARGET_SUBTYPE):
    # clip to [-1,1] before saving
    samples = np.clip(samples, -1.0, 1.0)
    sf.write(str(path), samples, sr, subtype=subtype)

def peak_normalize(samples):
    peak = np.max(np.abs(samples)) + EPS
    return samples / peak

def trim_silence(samples, sr=TARGET_SR, top_db=30):
    intervals = librosa.effects.split(samples, top_db=top_db, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH)
    if intervals.size == 0:
        return samples
    parts = [samples[s:e] for s, e in intervals]
    return np.concatenate(parts)

def spectral_subtract_noise_reduction(samples, sr=TARGET_SR, prop_decrease=0.8):
    if len(samples) < sr // 2:
        return samples
    stft = librosa.stft(samples, n_fft=1024, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_frames = max(1, int((0.5 * sr - WIN_LENGTH) // HOP_LENGTH))
    noise_mag = np.median(mag[:, :noise_frames], axis=1, keepdims=True)
    mag_denoised = np.maximum(mag - prop_decrease * noise_mag, 0.0)
    stft_denoised = mag_denoised * np.exp(1j * phase)
    y = librosa.istft(stft_denoised, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, length=len(samples))
    return y

def compute_log_mel(samples, sr=TARGET_SR, n_mels=N_MELS, fmin=20, fmax=7600, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    mel = librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=n_mels,
                                         hop_length=hop_length, win_length=win_length,
                                         fmin=fmin, fmax=fmax, power=1.0)
    return np.log(mel + 1e-6)

def compute_mfcc(samples, sr=TARGET_SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_mels=N_MELS):
    return librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=n_mfcc,
                                hop_length=hop_length, win_length=win_length, n_mels=n_mels)
