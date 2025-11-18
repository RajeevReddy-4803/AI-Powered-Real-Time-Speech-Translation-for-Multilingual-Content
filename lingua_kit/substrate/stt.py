from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import torchaudio

try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover - optional dependency
    sr = None

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from ..config import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_STT_METHOD,
    GOOGLE_LANGUAGE_MAP,
    TARGET_LANGS,
    WHISPER_MODEL_DIR,
)


LOGGER = logging.getLogger(__name__)


class WhisperBackend:
    def __init__(self, model_dir: Path = WHISPER_MODEL_DIR, language: str = "hi", task: str = "transcribe"):
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Whisper model directory '{model_dir}' was not found. "
                "Train the ASR model first via models/whisper/train_whisper_asr.py."
            )
        self.processor = WhisperProcessor.from_pretrained(str(model_dir), language=language, task=task)
        self.model = WhisperForConditionalGeneration.from_pretrained(str(model_dir))
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task=task)
        self.model.eval()

    def transcribe(self, wav_path: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> str:
        waveform, sr = torchaudio.load(str(wav_path))
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        waveform = waveform.squeeze(0).numpy()
        input_features = self.processor.feature_extractor(waveform, sampling_rate=sample_rate).input_features
        input_features = torch.tensor(input_features, dtype=torch.float32)
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, max_length=225)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()


class GoogleBackend:
    def __init__(self, max_chunk_seconds: int = 45):
        if sr is None:
            raise RuntimeError("speech_recognition is not installed. Run `pip install SpeechRecognition`.")
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.max_chunk_seconds = max_chunk_seconds

    def transcribe(self, wav_path: str | Path, hint_lang: Optional[str] = None) -> Optional[str]:
        audio_file = str(wav_path)
        language_code = GOOGLE_LANGUAGE_MAP.get(hint_lang, hint_lang)
        try:
            info = torchaudio.info(audio_file)
            total_duration = float(info.num_frames) / float(info.sample_rate)
        except Exception:
            total_duration = 0.0

        segments = []
        if total_duration and total_duration > self.max_chunk_seconds:
            offset = 0.0
            while offset < total_duration:
                duration = min(self.max_chunk_seconds, total_duration - offset)
                with sr.AudioFile(audio_file) as source:
                    audio_chunk = self.recognizer.record(source, duration=duration, offset=offset)
                text = self._recognize(audio_chunk, language_code)
                if text:
                    segments.append(text)
                offset += duration
        else:
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
            text = self._recognize(audio_data, language_code)
            if text:
                segments.append(text)

        combined = " ".join(seg.strip() for seg in segments if seg).strip()
        return combined or None

    def _recognize(self, audio_data, preferred_lang: Optional[str]) -> Optional[str]:
        fallback_chain = [
            preferred_lang,
            None,
            "en-US",
            "hi-IN",
        ]
        for lang in fallback_chain:
            if lang is None:
                func = lambda: self.recognizer.recognize_google(audio_data)
            else:
                func = lambda lang=lang: self.recognizer.recognize_google(audio_data, language=lang)
            try:
                return func()
            except Exception:
                continue
        return None


class SpeechToTextEngine:
    """
    Unified façade that exposes the Lingua Kit STT options but defaults to the fine-tuned Whisper model.
    """

    def __init__(
        self,
        method: str = DEFAULT_STT_METHOD,
        whisper_language: str = "hi",
        whisper_task: str = "transcribe",
    ):
        self.method = method.lower()
        self.whisper_language = whisper_language
        self.whisper_task = whisper_task
        self._whisper = None
        self._google = None

    @staticmethod
    def available_languages() -> dict[str, str]:
        return TARGET_LANGS

    def transcribe(self, wav_path: str | Path, language_hint: Optional[str] = None) -> Optional[str]:
        method = self.method

        if method == "whisper":
            return self._whisper_backend().transcribe(wav_path)
        if method == "google":
            return self._google_backend().transcribe(wav_path, language_hint)
        if method == "auto":
            try:
                return self._whisper_backend().transcribe(wav_path)
            except Exception as whisper_err:
                LOGGER.warning("Whisper backend failed (%s). Falling back to Google STT.", whisper_err)
                return self._google_backend().transcribe(wav_path, language_hint)
        raise ValueError(f"Unsupported STT method: {method}")

    def _whisper_backend(self) -> WhisperBackend:
        if self._whisper is None:
            self._whisper = WhisperBackend(
                model_dir=WHISPER_MODEL_DIR,
                language=self.whisper_language,
                task=self.whisper_task,
            )
        return self._whisper

    def _google_backend(self) -> GoogleBackend:
        if self._google is None:
            self._google = GoogleBackend()
        return self._google


