from __future__ import annotations

from pathlib import Path
from typing import Optional

from deep_translator import GoogleTranslator
from gtts import gTTS

from ..config import TARGET_LANGS


def translate_text(text: str, target_lang: str) -> Optional[str]:
    translator = GoogleTranslator(source="auto", target=target_lang)
    try:
        return translator.translate(text)
    except Exception as exc:  # pragma: no cover - network failures
        print(f"⚠️ Translation to {target_lang} failed: {exc}")
        return None


def synthesize_speech(
    text: str,
    lang_code: str,
    output_file: str | Path,
    fallback_lang: str = "hi",
) -> bool:
    """
    Save translated text as MP3 using gTTS. Falls back to Hindi voices for languages gTTS doesn't support.
    """
    lang = lang_code
    if lang_code not in AVAILABLE_GTTS_LANGS:
        lang = fallback_lang
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(str(output_file))
        return True
    except Exception as exc:
        print(f"⚠️ TTS failed for {lang_code}: {exc}")
        return False


def get_target_langs(override: Optional[str] = None) -> dict[str, str]:
    if override:
        selected = {}
        for code in override.split(","):
            code = code.strip()
            if code and code in TARGET_LANGS:
                selected[code] = TARGET_LANGS[code]
        return selected
    return TARGET_LANGS


# Minimal gTTS supported languages list — gTTS has >100 languages, but we'll special-case the
# languages this Lingua Kit stack supports.
AVAILABLE_GTTS_LANGS = {
    "en",
    "hi",
    "bn",
    "gu",
    "kn",
    "ml",
    "mr",
    "ta",
    "te",
    "ur",
}


