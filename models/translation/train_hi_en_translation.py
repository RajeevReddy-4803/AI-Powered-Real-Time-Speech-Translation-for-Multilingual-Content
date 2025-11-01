"""
Improved Hindi→English translation training (mT5-small)
- 10 epochs, lower LR, label smoothing, better eval
- Compatible with Kaggle T4 GPU
"""

import os
import pandas as pd
from datasets import Dataset
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from evaluate import load

BASE = "/kaggle/working/speech_translation"
DATA_DIR = f"{BASE}/data/translation_parallel"
MODEL_OUT = f"{BASE}/models/translation/hi2en_mt5_v2"

# Load dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"src": "hi_text", "tgt": "en_text"})
    df = df.dropna(subset=["hi_text", "en_text"])
    return Dataset.from_pandas(df[["hi_text", "en_text"]])

train_ds = load_data(f"{DATA_DIR}/train.csv")
val_ds = load_data(f"{DATA_DIR}/val.csv")
test_ds = load_data(f"{DATA_DIR}/test.csv")

# Load tokenizer & model
model_name = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# Preprocess
def preprocess(batch):
    inputs = ["translate Hindi to English: " + t for t in batch["hi_text"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(batch["en_text"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
val_ds = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)
test_ds = test_ds.map(preprocess, batched=True, remove_columns=test_ds.column_names)

# Metrics
bleu = load("sacrebleu")
chrf = load("chrf")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu_score = bleu.compute(predictions=preds, references=[[l] for l in labels])["score"]
    chrf_score = chrf.compute(predictions=preds, references=[[l] for l in labels])["score"]
    return {"bleu": bleu_score, "chrf": chrf_score}

# Training args
args = Seq2SeqTrainingArguments(
    output_dir=MODEL_OUT,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    num_train_epochs=10,
    predict_with_generate=True,
    logging_dir=f"{MODEL_OUT}/logs",
    logging_strategy="steps",
    logging_steps=50,
    save_total_limit=3,
    generation_max_length=128,
    fp16=True,
    label_smoothing_factor=0.1,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(MODEL_OUT)

print(f"✅ Model saved at: {MODEL_OUT}")

# Final evaluation
print("📊 Evaluating on test set...")
metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
print(f"✅ Test metrics: {metrics}")


