"""
🔥 Full Whisper-small fine-tuning on complete dataset
✅ Stable, resumable, and optimized for 2×T4 GPUs
✅ Automatic checkpoint resume
✅ Saves best & last checkpoints
"""

import os, torch, torchaudio, pandas as pd, numpy as np, evaluate
from datasets import Dataset, disable_caching
from dataclasses import dataclass
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ====== Config ======
MODEL_NAME = "openai/whisper-small"
BASE = "/kaggle/working/speech_translation"
TRAIN_CSV = f"{BASE}/data/processed/splits/train.csv"
VAL_CSV = f"{BASE}/data/processed/splits/val.csv"
TEST_CSV = f"{BASE}/data/processed/splits/test.csv"
OUTPUT_DIR = f"{BASE}/models/whisper_small_finetuned_full"

BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 3
LR = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using CUDA with {torch.cuda.device_count()} GPU(s):")
for i in range(torch.cuda.device_count()):
    print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")

disable_caching()

# ====== Load Data ======
for path in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)
print(f"📊 Data — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ====== Processor and Model ======
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="hi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.use_cache = False
model.to(device)

# ====== Datasets ======
train_ds = Dataset.from_pandas(train_df[["wav_path", "transcript"]])
val_ds = Dataset.from_pandas(val_df[["wav_path", "transcript"]])
test_ds = Dataset.from_pandas(test_df[["wav_path", "transcript"]])

# ====== Data Collator ======
@dataclass
class OnTheFlyCollator:
    processor: WhisperProcessor
    sampling_rate: int = 16000
    max_seconds: int = 30
    def __call__(self, batch):
        audios, texts = [], []
        target_len = self.sampling_rate * self.max_seconds
        for ex in batch:
            w, sr = torchaudio.load(ex["wav_path"])
            if w.ndim > 1:
                w = w.mean(dim=0)
            if sr != self.sampling_rate:
                w = torchaudio.functional.resample(w, sr, self.sampling_rate)
            a = w.numpy().astype(np.float32)
            if len(a) < target_len:
                a = np.pad(a, (0, target_len - len(a)))
            else:
                a = a[:target_len]
            audios.append(a)
            texts.append(str(ex["transcript"]))
        inputs = self.processor.feature_extractor(audios, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        input_features = inputs.input_features
        labels = self.processor.tokenizer(texts, padding=True, return_tensors="pt").input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"input_features": input_features, "labels": labels}

data_collator = OnTheFlyCollator(processor)

# ====== Metrics ======
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids = np.where(pred.label_ids == -100, processor.tokenizer.pad_token_id, pred.label_ids)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

# ====== Training ======
def run_training():
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        bf16=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        predict_with_generate=True,
        logging_dir=f"{OUTPUT_DIR}/logs",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # ====== Auto Resume ======
    last_ckpt = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            last_ckpt = sorted(checkpoints, key=os.path.getmtime)[-1]
            print(f"🔁 Resuming from last checkpoint: {last_ckpt}")

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("✅ Training complete.")

    print("📊 Evaluating on test set ...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print("✅ Test metrics:", test_metrics)

run_training()
