
import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ====== PATHS ======
MODEL_DIR = "/kaggle/working/speech_translation/models/whisper_small_finetuned/checkpoint-2355"
BASE_MODEL = "openai/whisper-small"
TEST_CSV = "/kaggle/working/speech_translation/data/processed/splits/test.csv"

# ====== LOAD MODEL & PROCESSOR ======
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained(BASE_MODEL)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()

# ====== LOAD TEST DATA ======
df = pd.read_csv(TEST_CSV)

# Normalize column names (standardize to Hugging Face format)
df = df.rename(columns={"wav_path": "audio", "transcript": "text"})

# Filter out invalid rows
df["text"] = df["text"].astype(str).fillna("").str.strip()
df = df[df["text"] != ""].reset_index(drop=True)
print(f"✅ Loaded {len(df)} valid test samples from {TEST_CSV}")

# Expand relative or Drive-based paths
base_dir = os.path.dirname(TEST_CSV)
df["audio"] = df["audio"].apply(
    lambda x: x if os.path.isabs(x) else os.path.join(base_dir, os.path.basename(str(x)))
)
df = df[df["audio"].apply(os.path.exists)].reset_index(drop=True)
print(f"🎧 {len(df)} audio files found and verified")

# ====== CONVERT TO DATASET ======
test_dataset = Dataset.from_pandas(df)

# ====== AUDIO PREPROCESSING ======
def preprocess(batch):
    path = batch["audio"]
    speech_array, sr = torchaudio.load(path)
    speech_array = speech_array.squeeze()
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech_array = resampler(speech_array)
    inputs = processor.feature_extractor(
        speech_array.numpy(), sampling_rate=16000, return_tensors="pt"
    )
    batch["input_features"] = inputs.input_features[0]
    return batch

print("🔄 Preprocessing audio...")
test_dataset = test_dataset.map(preprocess, remove_columns=["audio"])

# ====== METRICS ======
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

pred_texts, ref_texts = [], []

# ====== INFERENCE ======
print("🚀 Running inference...")
for example in tqdm(test_dataset, desc="Evaluating"):
    input_features = torch.tensor(example["input_features"]).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    pred_texts.append(transcription.strip().lower())
    ref_texts.append(str(example["text"]).strip().lower())


# ====== SANITY CHECK ======
pairs = [(p, r) for p, r in zip(pred_texts, ref_texts) if r.strip() != ""]
if not pairs:
    raise ValueError("❌ All reference transcripts are empty — cannot compute metrics.")
pred_texts, ref_texts = zip(*pairs)

# ====== COMPUTE METRICS ======
wer = wer_metric.compute(predictions=pred_texts, references=ref_texts)
cer = cer_metric.compute(predictions=pred_texts, references=ref_texts)

# Character-level metrics (rough proxy for symbol-level performance)
flat_true = "".join(ref_texts)
flat_pred = "".join(pred_texts)
min_len = min(len(flat_true), len(flat_pred))
flat_true, flat_pred = flat_true[:min_len], flat_pred[:min_len]

acc = accuracy_score(list(flat_true), list(flat_pred))
prec = precision_score(list(flat_true), list(flat_pred), average="macro", zero_division=0)
rec = recall_score(list(flat_true), list(flat_pred), average="macro", zero_division=0)
f1 = f1_score(list(flat_true), list(flat_pred), average="macro", zero_division=0)

# ====== DISPLAY RESULTS ======
print("\n📊 Whisper Evaluation Results")
print(f"✅ Word Error Rate (WER):       {wer:.4f}")
print(f"✅ Character Error Rate (CER):  {cer:.4f}")
print(f"✅ Accuracy:                    {acc:.4f}")
print(f"✅ Precision:                   {prec:.4f}")
print(f"✅ Recall:                      {rec:.4f}")
print(f"✅ F1 Score:                    {f1:.4f}")

print("\n✅ Evaluation completed successfully!")

