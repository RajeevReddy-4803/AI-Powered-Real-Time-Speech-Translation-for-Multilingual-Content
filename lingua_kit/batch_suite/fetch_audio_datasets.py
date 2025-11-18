"""
Fetch sample multilingual audio files from ai4bharat/indicvoices_r dataset.
This script downloads a small subset of audio samples for testing.

Note: Some datasets may require HuggingFace authentication.
Run: huggingface-cli login (or use notebook_login() in Python)

If you encounter torch DLL errors, try:
1. Restart your terminal/IDE
2. Or use: huggingface-cli download ai4bharat/indicvoices_r --repo-type dataset
"""
import os
import sys

# Try to import datasets, handle errors gracefully
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Warning: Could not import datasets library: {e}")
    print("💡 Trying alternative approach...")
    DATASETS_AVAILABLE = False

try:
    import soundfile as sf
    import numpy as np
except ImportError:
    print("❌ Missing required packages: soundfile, numpy")
    print("Install with: pip install soundfile numpy")
    sys.exit(1)

# Languages available in indicvoices_r dataset
AVAILABLE_LANGUAGES = [
    "Assamese", "Bengali", "Bodo", "Gujarati", "Hindi", "Kannada",
    "Malayalam", "Manipuri", "Marathi", "Odia", "Punjabi", "Tamil",
    "Telugu", "Urdu"
]

def fetch_samples(languages=None, num_samples_per_lang=2, output_dir="data"):
    """
    Fetch audio samples from indicvoices_r dataset.
    
    Args:
        languages: List of language names (e.g., ["Hindi", "Bengali"]). 
                  If None, fetches samples from all available languages.
        num_samples_per_lang: Number of samples to download per language
        output_dir: Directory to save audio files
    """
    if not DATASETS_AVAILABLE:
        print("\n❌ Cannot fetch: datasets library not available due to dependency issues.")
        print("\n💡 ALTERNATIVE SOLUTIONS:")
        print("1. Use HuggingFace CLI:")
        print("   huggingface-cli login")
        print("   huggingface-cli download ai4bharat/indicvoices_r --repo-type dataset")
        print("\n2. Download manually from:")
        print("   https://huggingface.co/datasets/ai4bharat/indicvoices_r")
        print("\n3. Add your own audio files to the 'data' folder")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    if languages is None:
        languages = AVAILABLE_LANGUAGES
    
    # Validate languages
    invalid_langs = [lang for lang in languages if lang not in AVAILABLE_LANGUAGES]
    if invalid_langs:
        print(f"⚠️ Invalid languages: {invalid_langs}")
        print(f"Available: {AVAILABLE_LANGUAGES}")
        languages = [lang for lang in languages if lang in AVAILABLE_LANGUAGES]
    
    if not languages:
        print("❌ No valid languages specified!")
        return
    
    print(f"📥 Fetching {len(languages)} language(s) from ai4bharat/indicvoices_r dataset...")
    print("Note: This may require HuggingFace authentication (run 'huggingface-cli login')\n")
    
    total_files = 0
    
    for lang in languages:
        try:
            print(f"🔍 Loading {lang} dataset...")
            dataset = load_dataset("ai4bharat/indicvoices_r", lang, split="train")
            
            # Get samples (limit to avoid downloading too much)
            num_samples = min(num_samples_per_lang, len(dataset))
            selected = dataset.select(range(num_samples))
            
            for i, sample in enumerate(selected):
                audio = sample.get("audio")
                if audio is None:
                    continue
                
                # Get audio array and sample rate
                audio_array = audio.get("array", None)
                sample_rate = audio.get("sampling_rate", 16000)
                
                if audio_array is None:
                    continue
                
                # Convert to numpy array if needed
                if not isinstance(audio_array, np.ndarray):
                    audio_array = np.array(audio_array)
                
                # Ensure proper format (mono, float32)
                if len(audio_array.shape) > 1:
                    audio_array = np.mean(audio_array, axis=1)  # Convert to mono
                
                # Normalize if needed
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                    if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                        audio_array = audio_array / np.max(np.abs(audio_array))
                
                # Create filename
                filename = f"sample_{lang.lower()}_{i}.wav"
                filepath = os.path.join(output_dir, filename)
                
                # Save audio file
                sf.write(filepath, audio_array, sample_rate)
                print(f"  ✅ Saved: {filepath}")
                total_files += 1
            
            print(f"  ✅ {lang}: {num_samples} sample(s) downloaded\n")
            
        except Exception as e:
            print(f"  ❌ Error loading {lang}: {e}\n")
            print(f"  💡 Tip: You may need to authenticate with HuggingFace:")
            print(f"     Run: huggingface-cli login\n")
    
    print(f"🎉 Download complete! Total files: {total_files}")
    print(f"📁 Files saved in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # Fetch samples from all 12 target languages
    # Language code to full name mapping
    languages_to_fetch = [
        "Hindi",      # hi
        "Punjabi",    # pa
        "Marathi",    # mr
        "Kannada",    # kn
        "Telugu",     # te
        "Tamil",      # ta
        "Gujarati",   # gu
        "Malayalam",  # ml
        "Bengali",    # bn
        "Odia",       # or
        "Urdu"        # ur
        # Note: English (en) is typically not in indicvoices_r as it's an Indian languages dataset
        # You can add your own English audio files if needed
    ]
    
    # Fetch 2 samples per language (you can increase this)
    fetch_samples(languages=languages_to_fetch, num_samples_per_lang=2)
