from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ASR_COMBINED_DIR = DATA_DIR / "asr_combined"
TRANSLATION_OUTPUT_DIR = DATA_DIR / "translation_outputs"
TRANSLATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL_DIR = BASE_DIR / "models" / "whisper" / "whisper_finetuned"

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_STT_METHOD = os.getenv("TRANSLATION_STT_BACKEND", "whisper")
DEFAULT_TARGET_LANGS = os.getenv("TRANSLATION_TARGET_LANGS")

TARGET_LANGS = {
    "en": "English",
    "hi": "Hindi",
    "pa": "Punjabi",
    "mr": "Marathi",
    "kn": "Kannada",
    "te": "Telugu",
    "ta": "Tamil",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "bn": "Bengali",
    "or": "Odia",
    "ur": "Urdu",
}

# Some components (e.g., Google Speech Recognition) need locale variants
GOOGLE_LANGUAGE_MAP = {
    "en": "en-US",
    "hi": "hi-IN",
    "pa": "pa-IN",
    "mr": "mr-IN",
    "kn": "kn-IN",
    "te": "te-IN",
    "ta": "ta-IN",
    "gu": "gu-IN",
    "ml": "ml-IN",
    "bn": "bn-IN",
    "or": "or-IN",
    "ur": "ur-PK",
}


