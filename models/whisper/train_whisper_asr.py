# models/whisper/train_whisper_asr.py
"""
Memory-optimized Whisper fine-tuning (on-the-fly audio, no .npy caching).
Colab T4 friendly: gradient checkpointing disabled (fixes backward errors),
fp16 mixed precision, 30s padding, small batch sizes.
"""

import os
from pathlib import Path
import pandas as pd
import torch
import torchaudio
import numpy as np
from datasets import Dataset, disable_caching
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import inspect

# ---------- Config ----------
MODEL_NAME = os.environ.get("WHISPER_MODEL", "openai/whisper-small")
LANGUAGE = "hi"
TASK = "transcribe"

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "splits"
OUTPUT_DIR = BASE_DIR / "models" / "whisper" / "whisper_finetuned_streamed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train_manifest.csv"
VAL_CSV = DATA_DIR / "val_manifest.csv"

disable_caching()

# ---------- Device ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = True

# ---------- Load dataframes ----------
if not TRAIN_CSV.exists() or not VAL_CSV.exists():
    raise FileNotFoundError(f"Train/Val CSVs not found at {TRAIN_CSV} and {VAL_CSV}")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
print(f"Loaded train={len(train_df)} val={len(val_df)}")

# Normalize column names
for df in (train_df, val_df):
    if "wav_path" not in df.columns:
        if "out_wav" in df.columns:
            df.rename(columns={"out_wav": "wav_path"}, inplace=True)
        elif "path" in df.columns:
            df.rename(columns={"path": "wav_path"}, inplace=True)
        else:
            raise KeyError("Manifest must contain 'wav_path' / 'out_wav' / 'path' column")

train_ds = Dataset.from_pandas(train_df[["wav_path", "transcript"]].reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df[["wav_path", "transcript"]].reset_index(drop=True))

# ---------- Processor & Model ----------
print("Loading processor + model:", MODEL_NAME)
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# Disable caching and checkpointing (prevents backward reuse errors)
model.config.use_cache = False
model.config.gradient_checkpointing = False
if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()

model.to(device)

# ---------- Audio Loader ----------
def load_audio_numpy(path, sr=16000):
    """Load and resample audio as float32 numpy."""
    waveform, orig_sr = torchaudio.load(path)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if int(orig_sr) != int(sr):
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    return waveform.cpu().numpy().astype(np.float32)

# ---------- Data Collator ----------
def data_collator(batch):
    """On-the-fly padding/cropping + feature extraction + label encoding."""
    wav_paths = [ex["wav_path"] for ex in batch]
    transcripts = [str(ex["transcript"]) for ex in batch]

    target_sr = 16000
    target_len = target_sr * 30  # 30s fixed input length
    audios = []

    for p in wav_paths:
        audio = load_audio_numpy(p, sr=target_sr)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        elif len(audio) > target_len:
            audio = audio[:target_len]
        audios.append(audio)

    inputs = processor.feature_extractor(audios, sampling_rate=target_sr, return_tensors="pt", padding=False)
    input_features = inputs["input_features"]

    labels_batch = processor.tokenizer(transcripts, padding=True, return_tensors="pt").input_ids
    labels = labels_batch.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {"input_features": input_features, "labels": labels}

# ---------- Metrics ----------
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    if isinstance(pred_ids, np.ndarray) and pred_ids.dtype != np.int64:
        pred_ids = np.argmax(pred_ids, axis=-1)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": 100 * wer, "cer": 100 * cer}

# ---------- Training Arguments ----------
training_args_dict = dict(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=4,          # safe for T4
    gradient_accumulation_steps=2,          # effective batch = 8
    learning_rate=1e-5,
    num_train_epochs=3,                     # balanced generalization
    fp16=torch.cuda.is_available(),         # saves VRAM
    save_strategy="epoch",
    evaluation_strategy="epoch" if "evaluation_strategy" in inspect.signature(Seq2SeqTrainingArguments).parameters else None,
    predict_with_generate=True,
    logging_dir=str(OUTPUT_DIR / "logs"),
    save_total_limit=2,
    dataloader_num_workers=0,               # avoid multiprocessing issues in Colab
    report_to="none",
    remove_unused_columns=False,
)

if training_args_dict.get("evaluation_strategy") is None:
    training_args_dict.pop("evaluation_strategy", None)

try:
    training_args = Seq2SeqTrainingArguments(**training_args_dict)
except TypeError:
    if "evaluation_strategy" in training_args_dict:
        training_args_dict["eval_strategy"] = training_args_dict.pop("evaluation_strategy")
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

# ---------- Trainer ----------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
)

# Resume training if checkpoint exists
last_ckpt = OUTPUT_DIR / "checkpoint-last"
resume = str(last_ckpt) if last_ckpt.exists() else None
if resume:
    print("Resuming from:", resume)

# ---------- Train ----------
print("Starting training ...")
trainer.train(resume_from_checkpoint=resume)

# ---------- Save ----------
trainer.save_model(str(OUTPUT_DIR))
processor.save_pretrained(str(OUTPUT_DIR))
print("✅ Training complete. Model saved at:", OUTPUT_DIR)
