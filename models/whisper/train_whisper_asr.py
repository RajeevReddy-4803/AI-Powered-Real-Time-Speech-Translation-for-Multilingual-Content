# models/whisper/train_whisper_asr.py
"""
Fine-tune OpenAI Whisper on Hindi + English ASR data (memory-safe version).
Handles low-memory systems by streaming features to disk.
Auto-detects transformers version for eval_strategy compatibility.
Supports resume-from-checkpoint for interrupted runs.
"""

import os
import pandas as pd
import torch
import torchaudio
import evaluate
import numpy as np
from pathlib import Path
from datasets import Dataset, disable_caching
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import inspect

# ---------- Config ----------
MODEL_NAME = "openai/whisper-large-v3"
LANGUAGE = "hi"
TASK = "transcribe"

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "asr_combined"
OUTPUT_DIR = BASE_DIR / "models" / "whisper" / "whisper_finetuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train_manifest.csv"
VAL_CSV = DATA_DIR / "val_manifest.csv"

disable_caching()

print(f"📂 Using data from: {DATA_DIR}")
print(f"💾 Model outputs will be saved to: {OUTPUT_DIR}")
print("🔍 GPU:", "✅ CUDA available" if torch.cuda.is_available() else "❌ CPU mode")

# ---------- Load Data ----------
try:
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    print(f"Train: {len(train_df)} samples | Val: {len(val_df)} samples")
except FileNotFoundError as e:
    print(f"❌ Error loading data files: {e}")
    print("Make sure you have run the preprocessing pipeline first.")
    exit(1)
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ---------- Load Processor & Model ----------
print("🔄 Loading Whisper processor & model ...")
try:
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    print("✅ Model and processor loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure you have internet connection to download the model.")
    exit(1)

# ---------- Helper ----------
def load_audio(path, sr=16000):
    """Load and resample audio file to target sample rate."""
    try:
        waveform, orig_sr = torchaudio.load(path)
        if orig_sr != sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
        return waveform.squeeze().numpy()
    except Exception as e:
        print(f"⚠️ Error loading audio {path}: {e}")
        return None

def prepare_example(example):
    """Prepare a single example for training."""
    audio = load_audio(example["out_wav"], sr=16000)
    if audio is None:
        raise ValueError(f"Failed to load audio for {example['out_wav']}")
    
    input_features = processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
    labels = processor.tokenizer(example["transcript"], return_tensors="pt").input_ids[0]
    return {"input_features": input_features, "labels": labels}

# ---------- Streamed Preprocessing ----------
def preprocess_stream(dataset, name):
    cache_dir = OUTPUT_DIR / f"cached_{name}"
    cache_dir.mkdir(exist_ok=True)
    npy_dir = cache_dir / "npy"
    npy_dir.mkdir(exist_ok=True)

    print(f"⚙️ Processing {name} dataset (streamed, memory-safe)...")
    records = []

    for i, ex in enumerate(dataset):
        processed = prepare_example(ex)
        idx = f"{name}_{i:06d}"

        feat_path = npy_dir / f"{idx}_feat.npy"
        lbl_path = npy_dir / f"{idx}_lbl.npy"
        np.save(feat_path, processed["input_features"])
        np.save(lbl_path, processed["labels"])

        records.append({"input_features": str(feat_path), "labels": str(lbl_path)})

        if (i + 1) % 500 == 0:
            chunk_file = cache_dir / f"{name}_chunk_{i//500}.csv"
            pd.DataFrame(records).to_csv(chunk_file, index=False)
            print(f"  Saved {i+1} samples → {chunk_file}")
            records = []

    if records:
        chunk_file = cache_dir / f"{name}_chunk_last.csv"
        pd.DataFrame(records).to_csv(chunk_file, index=False)
        print(f"  Saved last {len(records)} samples → {chunk_file}")

    csv_parts = list(cache_dir.glob("*.csv"))
    df = pd.concat([pd.read_csv(p) for p in csv_parts], ignore_index=True)
    return Dataset.from_pandas(df)


train_dataset = preprocess_stream(train_dataset, "train")
val_dataset = preprocess_stream(val_dataset, "val")

# ---------- Metrics ----------
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return {
        "wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    }

# ---------- Training Args (version-safe) ----------
args_kwargs = dict(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=10,
    fp16=torch.cuda.is_available(),
    save_strategy="epoch",
    predict_with_generate=True,
    logging_dir=str(OUTPUT_DIR / "logs"),
    generation_max_length=225,
    report_to="none",
    save_total_limit=3  # keep last 3 checkpoints
)

# Use correct arg name based on transformers version
if "evaluation_strategy" in inspect.signature(Seq2SeqTrainingArguments).parameters:
    args_kwargs["evaluation_strategy"] = "epoch"
else:
    args_kwargs["eval_strategy"] = "epoch"

training_args = Seq2SeqTrainingArguments(**args_kwargs)

# ---------- Data Collator ----------
def data_collator(batch):
    feats = [np.load(b["input_features"]) for b in batch]
    lbls = [np.load(b["labels"]) for b in batch]
    input_features = torch.tensor(feats, dtype=torch.float32)
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(l) for l in lbls],
        batch_first=True, padding_value=-100
    )
    return {"input_features": input_features, "labels": labels}

# ---------- Trainer ----------
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ---------- Resume Support ----------
checkpoint_dir = OUTPUT_DIR / "checkpoint-last"
if checkpoint_dir.exists():
    print(f"🔁 Resuming from previous checkpoint: {checkpoint_dir}")
    resume_from_checkpoint = str(checkpoint_dir)
else:
    resume_from_checkpoint = None

# ---------- Train ----------
print("🚀 Starting Whisper fine-tuning ...")
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

trainer.save_model(str(OUTPUT_DIR))
processor.save_pretrained(str(OUTPUT_DIR))
print("✅ Fine-tuning complete! Model saved to", OUTPUT_DIR)
