"""
Fine-tune OpenAI Whisper on the Hindi+English manifests emitted by the preprocessing pipeline.
Includes:
  • Automatic device detection (CPU/GPU)
  • Resume-from-checkpoint support
  • Configurable model size via env var WHISPER_MODEL_NAME
  • Text/Audio column auto-detection so manifests can evolve without code edits
"""

import os
from pathlib import Path
from typing import List

import evaluate
import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ---------------- PATH CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "asr_combined"
TRAIN_CSV = DATA_DIR / "train_manifest.csv"
VAL_CSV = DATA_DIR / "val_manifest.csv"
OUTPUT_DIR = BASE_DIR / "models" / "whisper" / "whisper_finetuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE = os.getenv("WHISPER_LANG", "hi")
TASK = os.getenv("WHISPER_TASK", "transcribe")
MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "openai/whisper-small")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(f"📂 Train manifest: {TRAIN_CSV}")
print(f"📂 Val manifest:   {VAL_CSV}")
print(f"💾 Output dir:     {OUTPUT_DIR}")


# ---------------- HELPERS ----------------
def detect_column(columns: List[str], candidates: List[str]) -> str:
    """Return the first column whose lowercase name contains any candidate token."""
    lowered = [c.lower() for c in columns]
    for token in candidates:
        for idx, name in enumerate(lowered):
            if token in name:
                return columns[idx]
    raise ValueError(f"Could not detect any of {candidates} within {columns}")


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    audio_col = detect_column(
        df.columns.tolist(),
        ["audio_filepath", "out_wav", "wav_path", "audio_path"],
    )
    text_col = detect_column(
        df.columns.tolist(),
        ["text", "transcript", "label"],
    )

    subset = df[[audio_col, text_col]].copy()
    subset.rename(columns={audio_col: "audio_path", text_col: "transcript"}, inplace=True)
    return subset


def load_manifest(csv_path: Path) -> Dataset:
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")
    frame = standardize_dataframe(pd.read_csv(csv_path))
    return Dataset.from_pandas(frame)


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze().numpy()


def prepare_example(batch):
    audio = load_audio(batch["audio_path"])
    input_features = processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
    labels = processor.tokenizer(batch["transcript"], return_tensors="pt").input_ids[0]
    batch["input_features"] = input_features
    batch["labels"] = labels
    return batch


# ---------------- LOAD DATA ----------------
train_dataset = load_manifest(TRAIN_CSV)
val_dataset = load_manifest(VAL_CSV)

# ---------------- LOAD MODEL ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
model.config.suppress_tokens = []
model.to(device)

train_dataset = train_dataset.map(prepare_example, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(prepare_example, remove_columns=val_dataset.column_names)

# ---------------- METRICS ----------------
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    return {
        "wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": 100 * cer_metric.compute(predictions=pred_str, references=label_str),
    }


# ---------------- TRAINING ARGS ----------------
fp16 = torch.cuda.is_available()
training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    num_train_epochs=12,
    warmup_steps=500,
    gradient_checkpointing=fp16,
    fp16=fp16,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=50,
    predict_with_generate=True,
    generation_max_length=225,
    save_total_limit=3,
    report_to="none",
)


def data_collator(batch):
    features = torch.tensor([b["input_features"] for b in batch], dtype=torch.float32)
    labels = [torch.tensor(b["labels"], dtype=torch.long) for b in batch]
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_features": features, "labels": labels}


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)

# ---------------- TRAIN / RESUME ----------------
last_checkpoint = None
checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"), key=os.path.getmtime)
if checkpoints:
    last_checkpoint = str(checkpoints[-1])
    print(f"♻️ Resuming from checkpoint: {last_checkpoint}")

print("🚀 Starting Whisper fine-tuning ...")
trainer.train(resume_from_checkpoint=last_checkpoint)

trainer.save_model(str(OUTPUT_DIR))
processor.save_pretrained(str(OUTPUT_DIR))
print(f"✅ Training complete. Model saved to {OUTPUT_DIR}")
