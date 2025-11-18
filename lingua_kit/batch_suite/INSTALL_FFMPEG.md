# Installing FFmpeg for MP3 Audio Processing

FFmpeg is required to convert MP3 audio files to WAV format for speech recognition.

## Quick Install Options:

### Option 1: Using winget (Windows Package Manager)
```powershell
winget install ffmpeg
```

### Option 2: Manual Download
1. Go to: https://www.gyan.dev/ffmpeg/builds/
2. Download "ffmpeg-release-essentials.zip"
3. Extract it to `C:\ffmpeg`
4. Add `C:\ffmpeg\bin` to your PATH environment variable
5. Restart your terminal

### Option 3: Using Chocolatey (if installed)
```powershell
choco install ffmpeg -y
```

## Verify Installation:
```powershell
ffmpeg -version
```

## After Installing:
Run the batch translator again:
```powershell
python module2_batch_translator.py
```

