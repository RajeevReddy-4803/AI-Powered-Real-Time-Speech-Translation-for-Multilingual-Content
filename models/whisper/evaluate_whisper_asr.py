# models/whisper/evaluate_whisper_asr.py
"""
Evaluate fine-tuned Whisper model on the test manifest.
Computes WER and CER automatically.
"""

import torch
import torchaudio
import pandas as pd
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
import numpy as np
from pathlib import Path


# ---------- Config ----------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "asr_combined"
MODEL_DIR = BASE_DIR / "models" / "whisper" / "whisper_finetuned"
TEST_CSV = DATA_DIR / "test_manifest.csv"

LANGUAGE = "hi"   # or "en" for English, change if testing one language specifically
TASK = "transcribe"

print(f"📂 Loading test data from: {TEST_CSV}")
print(f"🧠 Loading model from: {MODEL_DIR}")

# ---------- Load Model and Processor ----------
try:
    processor = WhisperProcessor.from_pretrained(MODEL_DIR, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    model.config.suppress_tokens = []
    model.eval()
    print("✅ Model and processor loaded successfully")
except Exception as e:
    print(f"❌ Error loading model from {MODEL_DIR}: {e}")
    print("Make sure the model has been trained and saved first.")
    exit(1)

# ---------- Load Test Data ----------
try:
    df = pd.read_csv(TEST_CSV)
    df = df.dropna(subset=["transcript"])
    dataset = Dataset.from_pandas(df)
    print(f"✅ Loaded {len(dataset)} test samples")
except FileNotFoundError as e:
    print(f"❌ Error loading test data from {TEST_CSV}: {e}")
    print("Make sure you have run the preprocessing pipeline and created test data.")
    exit(1)
except Exception as e:
    print(f"❌ Error loading test data: {e}")
    exit(1)

# ---------- Helper: Load Audio ----------
def load_audio(path, sr=16000):
    waveform, orig_sr = torchaudio.load(path)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    return waveform.squeeze().numpy()

# ---------- Evaluation Metrics ----------
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# ---------- Inference ----------
preds, refs = [], []
print("🎧 Running inference on test set (this may take time)...")

for i, example in enumerate(dataset):
    audio = load_audio(example["out_wav"], sr=16000)
    input_features = processor.feature_extractor(audio, sampling_rate=16000).input_features
    input_features = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_length=225)

    pred_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    ref_text = example["transcript"]

    preds.append(pred_text)
    refs.append(ref_text)

    if i % 50 == 0:
        print(f"→ Processed {i}/{len(dataset)} samples")

# ---------- Compute Metrics ----------
wer = 100 * wer_metric.compute(predictions=preds, references=refs)
cer = 100 * cer_metric.compute(predictions=preds, references=refs)

print("\n📊 --- Evaluation Results ---")
print(f"WER: {wer:.2f}%")
print(f"CER: {cer:.2f}%")

# ---------- Save Predictions ----------
out_path = MODEL_DIR / "test_predictions.csv"
pd.DataFrame({"reference": refs, "prediction": preds}).to_csv(out_path, index=False, encoding="utf-8")
print(f"✅ Saved detailed predictions to: {out_path}")
