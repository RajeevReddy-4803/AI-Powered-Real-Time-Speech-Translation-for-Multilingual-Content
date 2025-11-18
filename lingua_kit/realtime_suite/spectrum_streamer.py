"""
Realtime Suite — Low-latency speech-to-speech translator (Whisper + Deep Translator + gTTS).
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import librosa
from dotenv import load_dotenv

from lingua_kit.config import DEFAULT_STT_METHOD, TARGET_LANGS, TRANSLATION_OUTPUT_DIR
from lingua_kit.substrate.audio import ensure_wav
from lingua_kit.substrate.stt import SpeechToTextEngine
from lingua_kit.substrate.translate import synthesize_speech, translate_text

load_dotenv()

# ---------- CONFIG ----------
# Choose STT method: "whisper" (local, high quality) or "google" (cloud, free tier)
# Default to the shared configuration (fine-tuned Whisper)
STT_METHOD = os.getenv("STT_METHOD", DEFAULT_STT_METHOD)

# Target languages (shared with Module 2) now live in translation.config

# ---------- HELPERS ----------
DEFAULT_OUTPUT_DIR = TRANSLATION_OUTPUT_DIR / "realtime_outputs"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_dependencies(stt_method: str) -> bool:
    """Ensure the selected STT backend can be initialized."""
    issues = []
    try:
        get_stt_engine(stt_method)
    except Exception as exc:  # pragma: no cover - informative logging
        issues.append(f"❌ STT backend '{stt_method}' failed to initialize: {exc}")

    if issues:
        print("\n⚠️ Missing Dependencies:\n")
        for issue in issues:
            print(f"  {issue}")
        print("\n💡 Run: pip install -r requirements.txt")
        return False

    return True


def convert_to_wav(audio_file: str) -> Optional[Path]:
    """Convert audio file to WAV format (16kHz, mono) for processing."""
    try:
        return ensure_wav(audio_file)
    except Exception as exc:  # pragma: no cover - conversion errors depend on codecs
        print(f"⚠️ Audio conversion failed: {exc}")
        return None


@lru_cache(maxsize=3)
def get_stt_engine(method: Optional[str]) -> SpeechToTextEngine:
    resolved = (method or STT_METHOD).lower()
    return SpeechToTextEngine(method=resolved)


def text_to_speech(text: str, lang_code: str, output_file: Path) -> bool:
    """Wrapper around shared TTS helper for backwards compatibility."""
    return synthesize_speech(text, lang_code, output_file)

def realtime_translate_audio_file(
    audio_file: str,
    target_lang: str = "hi",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    stt_method: str = None,
    save_output: bool = True
) -> Dict:
    """
    Complete pipeline: Translate audio file in real-time.
    
    Args:
        audio_file: Input audio file path
        target_lang: Target language code
        output_dir: Output directory for translated audio
        stt_method: STT method ("whisper" or "google")
        save_output: If True, save TTS audio files
    
    Returns:
        Dictionary with translation results
    """
    print("="*60)
    print("Realtime Suite — Lingua Kit")
    print("="*60)
    print()
    
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    source_audio = Path(audio_file)

    stt_method = stt_method or STT_METHOD
    if not check_dependencies(stt_method):
        return None
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Input: {os.path.basename(audio_file)}")
    print(f"🎯 Target Language: {TARGET_LANGS.get(target_lang, target_lang)} ({target_lang})")
    print(f"🎤 STT Method: {stt_method.upper()}")
    print()
    
    # Convert to WAV if needed
    print("🔄 Converting audio to WAV format...")
    wav_path = convert_to_wav(audio_file)
    if not wav_path:
        return None
    
    # Speech to Text
    print(f"\n📝 Step 1: Speech-to-Text ({stt_method})...")
    engine = get_stt_engine(stt_method)
    text = engine.transcribe(wav_path)
    if not text:
        print("❌ Speech recognition failed")
        return None
    
    print(f"💬 Recognized Text: {text}")
    print()
    
    # Translate
    print(f"🌐 Step 2: Translation to {TARGET_LANGS.get(target_lang, target_lang)}...")
    translated = translate_text(text, target_lang)
    if not translated:
        print("❌ Translation failed")
        return None
    
    print(f"🌐 Translated Text: {translated}")
    print()
    
    output_file_path: Optional[Path] = None

    # Text to Speech
    if save_output:
        print(f"🔊 Step 3: Text-to-Speech...")
        output_file_path = output_dir / f"{source_audio.stem}_{target_lang}.mp3"
        if text_to_speech(translated, target_lang, output_file_path):
            print(f"✅ Saved: {output_file_path}")
        else:
            print("⚠️ TTS generation failed")
    
    result = {
        "input_file": str(source_audio),
        "recognized_text": text,
        "translated_text": translated,
        "target_language": target_lang,
        "output_file": str(output_file_path) if output_file_path else None,
    }
    
    print()
    print("="*60)
    print("✅ Realtime Suite translation complete!")
    print("="*60)
    
    if wav_path != source_audio and wav_path.exists():
        try:
            wav_path.unlink()
        except OSError:
            pass

    return result

def batch_realtime_translate(
    audio_files: List[str],
    target_langs: Optional[List[str]] = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    stt_method: str = None,
) -> List[Dict]:
    """
    Batch translate multiple audio files to multiple languages.
    
    Args:
        audio_files: List of input audio file paths
        target_langs: List of target language codes (default: all 12 languages)
        output_dir: Output directory
        stt_method: STT method
    
    Returns:
        List of translation results
    """
    if target_langs is None:
        target_langs = list(TARGET_LANGS.keys())
    
    results = []
    
    print(f"\n📊 Batch Translation: {len(audio_files)} files → {len(target_langs)} languages")
    print(f"📈 Total: {len(audio_files) * len(target_langs)} outputs\n")
    
    from tqdm import tqdm
    
    for audio_file in tqdm(audio_files, desc="Processing files"):
        for target_lang in target_langs:
            result = realtime_translate_audio_file(
                audio_file=audio_file,
                target_lang=target_lang,
                output_dir=output_dir,
                stt_method=stt_method,
                save_output=True,
            )
            if result:
                results.append(result)
    
    return results

# ---------- MAIN ----------
if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        target_lang = sys.argv[2] if len(sys.argv) > 2 else "hi"
        
        realtime_translate_audio_file(
            audio_file=audio_file,
            target_lang=target_lang,
            stt_method=STT_METHOD
        )
    else:
        print("💡 Usage:")
        print("  python module3_ott_realtime.py <audio_file> [target_lang]")
        print()
        print("  Example:")
        print("    python module3_ott_realtime.py ../module2/data/sample_en_000.mp3 hi")
        print()
        print("📚 Supported languages:")
        for code, name in TARGET_LANGS.items():
            print(f"    {code}: {name}")
