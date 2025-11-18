"""
Web Portal — Flask interface for the Lingua Kit toolkit.
Upload audio/video files or use microphone for real-time translation.
"""
from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
from pathlib import Path
from werkzeug.utils import secure_filename
import re
import subprocess
import numpy as np
import sys

# Audio processing
import librosa
import soundfile as sf
from scipy import signal
from pydub import AudioSegment

# Optional YouTube imports (lazy-loaded where possible)
try:
    from pytube import YouTube
    PYTUBE_AVAILABLE = True
except Exception:
    PYTUBE_AVAILABLE = False

# Edge TTS for gender-matched voices
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# Shared translation stack
from lingua_kit.config import DEFAULT_STT_METHOD, TARGET_LANGS, TRANSLATION_OUTPUT_DIR
from lingua_kit.substrate.audio import ensure_wav
from lingua_kit.substrate.stt import SpeechToTextEngine
from lingua_kit.substrate.translate import translate_text as shared_translate_text, synthesize_speech


# ---------- CONFIG ----------
app = Flask(__name__)

# Ensure console can print unicode (emojis) without crashing on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
WEBAPP_DATA_DIR = TRANSLATION_OUTPUT_DIR / "web_app"
UPLOAD_DIR = WEBAPP_DATA_DIR / "uploads"
STATIC_DIR = Path(__file__).parent / "static"

app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['STATIC_FOLDER'] = str(STATIC_DIR)

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Configure pydub to use embedded ffmpeg if available (avoids system ffmpeg requirement)
try:
    import imageio_ffmpeg
    _ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    # pydub looks at AudioSegment.converter; set it to the ffmpeg binary
    if _ffmpeg_bin and os.path.exists(_ffmpeg_bin):
        AudioSegment.converter = _ffmpeg_bin
        # Some pydub versions also read ffmpeg/ffprobe attributes
        try:
            from pydub.utils import which
            # Overwrite which to return embedded ffmpeg path when asked for ffmpeg/ffprobe
            os.environ['FFMPEG_BINARY'] = _ffmpeg_bin
        except Exception:
            pass
        print(f"🎛 Using embedded ffmpeg for pydub: {_ffmpeg_bin}")
except Exception as _e:
    print(f"⚠️ Could not configure embedded ffmpeg: {_e}")

# Shared language catalog imported from lingua_kit.config

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'ogg', 'mp4', 'avi', 'mov', 'mkv'}

WEBAPP_STT_BACKEND = os.getenv("WEBAPP_STT_BACKEND", DEFAULT_STT_METHOD)
STT_ENGINE = SpeechToTextEngine(method=WEBAPP_STT_BACKEND)

# ---------- HELPERS ----------
def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(audio_file):
    """Convert audio file to WAV format (16kHz, mono) for speech recognition."""
    try:
        wav_path = ensure_wav(audio_file)
        return str(wav_path)
    except Exception as exc:
        print(f"❌ Audio conversion failed: {exc}")
        return None

def speech_to_text(audio_file, source_language=None, max_chunk_seconds: int = 45):
    """Convert speech to text using the shared SpeechToTextEngine (Whisper or Google)."""
    try:
        return STT_ENGINE.transcribe(audio_file, language_hint=source_language)
    except Exception as exc:
        print(f"❌ STT failed: {exc}")
        return None

def translate_text(text, target_lang):
    """Translate text to target language using shared helper."""
    try:
        return shared_translate_text(text, target_lang)
    except Exception as e:
        print(f"⚠️ Translation failed: {e}")
        return None

def detect_gender_from_audio(audio_file):
    """
    Detect speaker gender from audio using pitch analysis.
    Returns 'male', 'female', or 'unknown'
    """
    try:
        # Load audio (analyze first 5 seconds for faster processing)
        y, sr = librosa.load(audio_file, sr=16000, mono=True, duration=5.0)
        
        if len(y) < sr:  # Too short (< 1 second)
            return 'unknown'
        
        # Use librosa's pyin (probabilistic YIN) for better pitch detection
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz (lowest male)
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz (highest female)
            frame_length=2048
        )
        
        # Filter out unvoiced frames and get valid pitch values
        pitch_values = f0[voiced_flag]
        
        if len(pitch_values) == 0:
            # Fallback: try autocorrelation method
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0 and 80 < pitch < 400:  # Valid human pitch range
                    pitch_values.append(pitch)
        
        if len(pitch_values) == 0:
            return 'unknown'
        
        # Calculate statistics
        avg_pitch = np.mean(pitch_values)
        median_pitch = np.median(pitch_values)
        
        # Use median for more robust classification (less affected by outliers)
        final_pitch = median_pitch
        
        # Gender classification based on pitch
        # Male voices typically: 85-180 Hz (average ~120 Hz)
        # Female voices typically: 165-255 Hz (average ~220 Hz)
        # There's overlap around 150-200 Hz, use median for better accuracy
        if final_pitch < 140:
            return 'male'
        elif final_pitch > 200:
            return 'female'
        else:
            # Ambiguous range (140-200 Hz) - use average as tie-breaker
            if avg_pitch < 170:
                return 'male'
            else:
                return 'female'
    
    except Exception as e:
        print(f"⚠️ Gender detection failed: {e}")
        return 'male'  # Default to male if detection fails

def get_edge_tts_voice(lang_code, gender='male'):
    """
    Get appropriate Edge TTS voice for language and gender.
    Returns voice name or None if not available.
    """
    if not EDGE_TTS_AVAILABLE:
        return None
    
    # Edge TTS voice mapping by language and gender
    # Format: {lang_code: {'male': 'voice_name', 'female': 'voice_name'}}
    voice_map = {
        'en': {
            'male': 'en-US-GuyNeural',  # or 'en-US-DavisNeural'
            'female': 'en-US-AriaNeural'
        },
        'hi': {
            'male': 'hi-IN-MadhurNeural',
            'female': 'hi-IN-SwaraNeural'
        },
        'mr': {
            'male': 'mr-IN-ManoharNeural',
            'female': 'mr-IN-AarohiNeural'
        },
        'ta': {
            'male': 'ta-IN-ValluvarNeural',
            'female': 'ta-IN-PallaviNeural'
        },
        'te': {
            'male': 'te-IN-MohanNeural',
            'female': 'te-IN-ShrutiNeural'
        },
        'kn': {
            'male': 'kn-IN-GaganNeural',
            'female': 'kn-IN-SapnaNeural'
        },
        'gu': {
            'male': 'gu-IN-NiranjanNeural',
            'female': 'gu-IN-DhwaniNeural'
        },
        'ml': {
            'male': 'ml-IN-MidhunNeural',
            'female': 'ml-IN-SobhanaNeural'
        },
        'bn': {
            'male': 'bn-IN-BashkarNeural',
            'female': 'bn-IN-TanishaaNeural'
        },
        'ur': {
            'male': 'ur-PK-AsadNeural',
            'female': 'ur-PK-UzmaNeural'
        },
        'pa': {
            'male': 'pa-IN-GurpreetNeural',
            'female': 'pa-IN-GurpreetNeural'  # Limited options
        },
        'or': {
            'male': 'or-IN-LekhaNeural',  # Limited options
            'female': 'or-IN-LekhaNeural'
        }
    }
    
    lang_voices = voice_map.get(lang_code)
    if lang_voices:
        return lang_voices.get(gender, lang_voices.get('male'))  # Default to male if gender not found
    
    return None

def text_to_speech(text, lang_code, output_path, gender='male'):
    """
    Convert text to speech with gender matching.
    Uses edge-tts if available (better voices), falls back to gTTS.
    """
    # Try Edge TTS first (better quality, gender-matched voices)
    if EDGE_TTS_AVAILABLE:
        try:
            voice = get_edge_tts_voice(lang_code, gender)
            if voice:
                print(f"🔊 Using Edge TTS voice: {voice} ({gender})")
                # Generate audio using edge-tts
                import asyncio
                import edge_tts
                
                async def generate():
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(output_path)
                
                # Run async function
                try:
                    asyncio.run(generate())
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        return True
                except Exception as e:
                    print(f"⚠️ Edge TTS failed: {e}, falling back to gTTS")
        except Exception as e:
            print(f"⚠️ Edge TTS error: {e}, falling back to gTTS")
    
    # Fallback to shared gTTS helper (handles unsupported languages gracefully)
    print("🔊 Using shared gTTS fallback")
    return synthesize_speech(text, lang_code, output_path)


# ---------- ROUTES ----------
@app.route('/')
def index():
    """Main page."""
    return render_template('index.html', languages=TARGET_LANGS)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and translation."""
    try:
        # Check if file is present (accept both 'file' and 'audio')
        if 'file' in request.files:
            file = request.files['file']
        elif 'audio' in request.files:
            file = request.files['audio']
        else:
            return jsonify({"error": "No file provided (expected form field 'file' or 'audio')"}), 400
        target_lang = request.form.get('lang', 'hi')
        gender = request.form.get('gender', 'male')  # Get gender from user selection
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        print(f"📁 Uploaded: {filename}")
        
        # Convert to WAV for speech recognition
        wav_file = convert_to_wav(file_path)
        if not wav_file:
            return jsonify({"error": "Failed to convert audio file"}), 500
        
        # Speech to Text (with improved accuracy)
        # Try to detect source language from file name if available
        source_lang = None
        for lang_code in TARGET_LANGS.keys():
            if lang_code in filename.lower():
                source_lang = lang_code
                break
        
        original_text = speech_to_text(wav_file, source_language=source_lang)
        if not original_text:
            return jsonify({"error": "Could not recognize speech. Please ensure audio contains clear speech."}), 400
        
        print(f"💬 Recognized: {original_text}")
        
        # Use user-selected gender (not auto-detection)
        print(f"👤 Using selected voice gender: {gender}")
        
        # Translate
        translated_text = translate_text(original_text, target_lang)
        if not translated_text:
            return jsonify({"error": "Translation failed"}), 500
        
        print(f"🌐 Translated: {translated_text}")
        
        # Generate TTS with user-selected gender
        tts_filename = f"translated_{target_lang}_{Path(filename).stem}.mp3"
        tts_path = os.path.join(app.config['STATIC_FOLDER'], tts_filename)
        if not text_to_speech(translated_text, target_lang, tts_path, gender=gender):
            return jsonify({"error": "TTS generation failed"}), 500
        
        # Clean up temporary WAV file
        if os.path.exists(wav_file):
            os.remove(wav_file)
        
        return jsonify({
            "success": True,
            "original_text": original_text,
            "translated_text": translated_text,
            "target_language": target_lang,
            "audio_url": f"/static/{tts_filename}"
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/translate_text', methods=['POST'])
def translate_text_only():
    """Translate text directly (for real-time mic input)."""
    try:
        data = request.json
        text = data.get('text', '').strip()
        target_lang = data.get('lang', 'hi')
        gender = data.get('gender', 'male')  # Get gender from request, default to male
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Translate
        translated_text = translate_text(text, target_lang)
        if not translated_text:
            return jsonify({"error": "Translation failed"}), 500
        
        # Generate TTS with gender matching
        tts_filename = f"translated_live_{target_lang}.mp3"
        tts_path = os.path.join(app.config['STATIC_FOLDER'], tts_filename)
        if text_to_speech(translated_text, target_lang, tts_path, gender=gender):
            return jsonify({
                "success": True,
                "translated_text": translated_text,
                "audio_url": f"/static/{tts_filename}?t={int(os.path.getmtime(tts_path))}"
            })
        else:
            return jsonify({
                "success": True,
                "translated_text": translated_text,
                "audio_url": None
            })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/static/<filename>')
def static_file(filename):
    """Serve static files."""
    return send_file(os.path.join(app.config['STATIC_FOLDER'], filename))


@app.route('/mic_record', methods=['POST'])
def mic_record():
    """Accept microphone audio upload (field name 'audio') and translate + TTS."""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio provided"}), 400

        file = request.files['audio']
        target_lang = request.form.get('lang', 'hi')
        gender = request.form.get('gender', 'male')

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename or 'mic_audio.wav')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        print(f"🎙 Received mic audio: {filename}")

        wav_file = convert_to_wav(file_path)
        if not wav_file:
            return jsonify({"error": "Failed to convert audio file"}), 500

        # Allow optional explicit source language for improved accuracy
        source_lang_hint = request.form.get('source_lang')
        original_text = speech_to_text(wav_file, source_language=source_lang_hint)
        if not original_text:
            return jsonify({"error": "Could not recognize speech."}), 400

        translated_text = translate_text(original_text, target_lang)
        if not translated_text:
            return jsonify({"error": "Translation failed"}), 500

        tts_filename = f"translated_live_{target_lang}.mp3"
        tts_path = os.path.join(app.config['STATIC_FOLDER'], tts_filename)
        if not text_to_speech(translated_text, target_lang, tts_path, gender=gender):
            return jsonify({"error": "TTS generation failed"}), 500

        try:
            if os.path.exists(wav_file):
                os.remove(wav_file)
        except Exception:
            pass

        return jsonify({
            "success": True,
            "original_text": original_text,
            "translated_text": translated_text,
            "audio_url": f"/static/{tts_filename}?t={int(os.path.getmtime(tts_path))}"
        })
    except Exception as e:
        print(f"❌ Mic processing error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------- YOUTUBE TRANSLATION (CHUNKED) ----------
def download_youtube_audio_to_file(url: str) -> str:
    """
    Download YouTube audio-only stream and return a local file path.
    Returns the path to the downloaded file (usually .mp4/m4a).
    """
    if not PYTUBE_AVAILABLE:
        raise RuntimeError("pytube is not installed. Please install dependencies.")
    # Prefer pytube; fall back to yt_dlp for better compatibility
    # Returns path to downloaded file; conversion is handled later by convert_to_wav
    last_err = None
    try:
        if not (url.startswith('http://') or url.startswith('https://')):
            raise RuntimeError("Invalid URL. Include http(s) scheme.")
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        if stream is None:
            raise RuntimeError("No audio stream available for this YouTube URL (private/age/region restricted?).")
        out_file = stream.download(filename="yt_audio")
        return out_file
    except Exception as e:
        last_err = e

    # Fallback: yt-dlp
    try:
        import yt_dlp
        tmp_base = os.path.join(tempfile.gettempdir(), 'yt_audio')
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': tmp_base + '.%(ext)s',
            'quiet': True,
            'noplaylist': True,
            'nocheckcertificate': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0',
            },
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            out_file = ydl.prepare_filename(info)
        if not os.path.exists(out_file):
            raise RuntimeError("yt-dlp did not produce a file")
        return out_file
    except Exception as e:
        raise RuntimeError(f"YouTube download failed: {e if e else last_err}")


def youtube_transcribe_chunk(wav_path: str, start_ms: int, end_ms: int) -> str:
    """
    Slice the WAV using librosa (no ffmpeg required) and run existing speech_to_text.
    """
    duration_s = max(0.01, (end_ms - start_ms) / 1000.0)
    offset_s = max(0.0, start_ms / 1000.0)
    tmp_chunk = tempfile.mktemp(suffix=".wav")
    try:
        # Load just the needed window and save
        y, sr = librosa.load(wav_path, sr=16000, mono=True, offset=offset_s, duration=duration_s)
        if y.size == 0:
            return ""
        sf.write(tmp_chunk, y, 16000)
        text = speech_to_text(tmp_chunk)
        return (text or "").strip()
    except Exception as e:
        print(f"⚠️ Chunk STT failed {start_ms}-{end_ms}ms: {e}")
        return ""
    finally:
        try:
            os.remove(tmp_chunk)
        except Exception:
            pass


@app.route('/youtube_translate', methods=['POST'])
def youtube_translate():
    """
    Accepts JSON { url, lang [, gender] }.
    Downloads audio, chunks into 8s segments, transcribes with Whisper,
    translates each chunk, and generates per-chunk TTS.
    Returns list of chunk dicts.
    """
    try:
        # Accept JSON or form submissions
        data = request.get_json(silent=True) or {}
        if not data:
            data = request.form.to_dict() if request.form else {}
        youtube_url = (data.get('url') or '').strip()
        target_lang = (data.get('lang') or 'hi').strip()
        gender = (data.get('gender') or 'male').strip()

        if not youtube_url:
            return jsonify({"error": "Please provide a valid YouTube URL in the 'url' field."}), 400

        # Download audio file and convert to wav using existing pipeline
        downloaded_path = download_youtube_audio_to_file(youtube_url)
        wav_path = convert_to_wav(downloaded_path)
        if not wav_path:
            return jsonify({"error": "Failed to prepare audio from YouTube"}), 500

        # Chunk in 8s windows (derive duration with librosa to avoid ffmpeg)
        chunk_ms = 8000
        try:
            total_duration_s = librosa.get_duration(path=wav_path)
        except Exception:
            # Fallback: load fully then compute length
            y_full, sr_full = librosa.load(wav_path, sr=16000, mono=True)
            total_duration_s = len(y_full) / float(sr_full or 16000)
        total_ms = int(total_duration_s * 1000)
        total_ms = min(total_ms, 60000)
        results = []

        for start in range(0, total_ms, chunk_ms):
            end = min(start + chunk_ms, total_ms)
            print(f"🎧 Processing chunk {start//1000}s - {end//1000}s")
            text = youtube_transcribe_chunk(wav_path, start, end)
            if text:
                # Translate using existing translator for consistency
                translated_text = translate_text(text, target_lang) or ""
                # Generate TTS per chunk
                tts_filename = f"yt_chunk_{start}.mp3"
                tts_path = os.path.join(app.config['STATIC_FOLDER'], tts_filename)
                audio_ok = text_to_speech(translated_text or " ", target_lang, tts_path, gender=gender)
                results.append({
                    "start": start // 1000,
                    "original": text,
                    "translated": translated_text,
                    "audio": f"/static/{tts_filename}" if audio_ok else None
                })

        # Cleanup temp files
        try:
            if os.path.exists(wav_path) and wav_path != downloaded_path:
                os.remove(wav_path)
        except Exception:
            pass
        try:
            if os.path.exists(downloaded_path):
                os.remove(downloaded_path)
        except Exception:
            pass

        return jsonify(results)
    except Exception as e:
        print(f"❌ YouTube processing error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("🎧 Module 4 — Flask Real-Time Speech Translator")
    print("="*60)
    print("🌐 Starting server...")
    print("📱 Open: http://127.0.0.1:5000")
    print("="*60)
    app.run(debug=True, host='127.0.0.1', port=5000)

