# Speech Translation Project

A comprehensive speech preprocessing and automatic speech recognition (ASR) pipeline for Hindi and English audio data, featuring fine-tuned Whisper models.

## 🚀 Features

- **Multi-language Support**: Hindi and English audio processing
- **Advanced Preprocessing**: Noise reduction, silence trimming, normalization
- **Feature Extraction**: Log-mel spectrograms and MFCC features
- **Whisper Fine-tuning**: Memory-safe training with checkpoint support
- **Comprehensive Evaluation**: WER and CER metrics
- **Data Pipeline**: Automated manifest generation and dataset splitting

> Maintained by **Karukonda Rajeev Reddy**

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- At least 16GB RAM for training

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd speech_translation

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
speech_translation/
├── preprocess/           # Audio preprocessing pipeline
│   ├── preprocess_speech.py
│   ├── manifest_utils.py
│   ├── combine_manifests.py
│   ├── text_normalizer.py
│   └── utils_audio.py
├── models/whisper/       # Whisper model training and evaluation
│   ├── train_whisper_asr.py
│   └── evaluate_whisper_asr.py
├── lingua_kit/           # Lingua Kit translation + deployment stack
│   ├── baseline_lab/     # Environment + text translation sanity checks
│   ├── batch_suite/      # Multilingual batch speech translator
│   ├── realtime_suite/   # Low-latency OTT translator
│   ├── pulse_portal/     # Flask web UI + APIs
│   ├── ops_suite/        # End-to-end orchestration helpers
│   └── substrate/        # Shared audio/STT/translation helpers
├── data/                 # Processed datasets (gitignored)
├── raw_data/             # Original audio files (gitignored)
└── requirements.txt      # Dependencies
```

## 🔄 Usage Pipeline

### 1. Audio Preprocessing

#### Hindi Audio Processing
```bash
python -m preprocess.preprocess_speech \
  --input_dir ./raw_data/hindi \
  --output_dir ./data/hindi_preprocessed \
  --lang hi \
  --n_jobs 8 \
  --denoise --silence_db 30
```

#### English Audio Processing
```bash
python -m preprocess.preprocess_speech \
  --input_dir ./raw_data/english \
  --output_dir ./data/english_preprocessed \
  --lang en \
  --n_jobs 8 \
  --denoise --silence_db 30
```

**Outputs per file:**
- Processed audio: `<name>_proc.wav`
- Features: `logmel.npy`, `mfcc.npy`
- Normalized transcript: `<name>.txt`
- Manifest CSV: `preprocess_manifest.csv`

### 2. Exploratory Data Analysis (EDA)

```bash
# Hindi EDA
python -m preprocess.eda_preprocessed \
  --manifest ./data/hindi_preprocessed/preprocess_manifest.csv \
  --out_dir ./data/hindi_preprocessed/eda \
  --num_examples 6

# English EDA
python -m preprocess.eda_preprocessed \
  --manifest ./data/english_preprocessed/preprocess_manifest.csv \
  --out_dir ./data/english_preprocessed/eda \
  --num_examples 6
```

**EDA Outputs:**
- `duration_distributions.png` - Audio duration statistics
- `duration_stats.csv` - Detailed duration metrics
- `random_logmel_examples.png` - Sample spectrograms

### 3. Dataset Preparation

#### Create Train/Val/Test Splits
```bash
# Hindi dataset
python -m preprocess.manifest_utils \
  --manifest ./data/hindi_preprocessed/preprocess_manifest.csv \
  --out_dir ./data/hindi_preprocessed \
  --val_frac 0.05 --test_frac 0.05

# English dataset
python -m preprocess.manifest_utils \
  --manifest ./data/english_preprocessed/preprocess_manifest.csv \
  --out_dir ./data/english_preprocessed \
  --val_frac 0.05 --test_frac 0.05
```

#### Combine Datasets
```bash
python -m preprocess.combine_manifests
```

This creates combined train/val/test manifests in `./data/asr_combined/`.

### 4. Model Training

```bash
python models/whisper/train_whisper_asr.py
```

**Training Features:**
- Memory-safe streaming preprocessing
- Automatic checkpoint resumption
- GPU/CPU auto-detection
- Progress tracking and metrics

### 5. Model Evaluation

```bash
python models/whisper/evaluate_whisper_asr.py
```

**Evaluation Outputs:**
- WER (Word Error Rate) and CER (Character Error Rate)
- Detailed predictions: `test_predictions.csv`

### 6. Translation & Deployment Modules (Lingua Kit)

The Lingua Kit toolchain lives under `lingua_kit/` and reuses the fine-tuned Whisper checkpoints by default.

| Component | Description | Example command |
|-----------|-------------|-----------------|
| Baseline Lab | Text-only translation smoke test (Helsinki hi↔en) | `python lingua_kit/baseline_lab/lingua_probe.py` |
| Batch Suite | Multilingual batch speech translator (12+ languages, speech→speech) | `python lingua_kit/batch_suite/nimbus_batcher.py --input_dir ./data/english_preprocessed/wav` |
| Realtime Suite | Real-time OTT translator for files or batches | `python lingua_kit/realtime_suite/spectrum_streamer.py path/to/audio.mp3 hi` |
| Pulse Portal | Flask web app for uploads, microphone, and YouTube URLs | `python lingua_kit/pulse_portal/aurora_gateway.py` *(then open http://127.0.0.1:5000)* |
| Ops Suite | Turn-key script that chains preprocessing, Whisper training, evaluation, and sample translations | `python lingua_kit/ops_suite/atlas_driver.py` |

Supporting utilities (dataset downloaders, ffmpeg helpers, etc.) now live under `lingua_kit/batch_suite` and the shared audio/STT/translation helpers in `lingua_kit/substrate/` are imported across every module. All Lingua Kit outputs (logs, synthesized audio, temporary uploads) are written beneath `data/translation_outputs/` to keep Git clean.

Batch Suite and Realtime Suite default to the Whisper backend; pass `--stt_backend google` (or set `WEBAPP_STT_BACKEND`) to fall back to the lightweight Google STT workflow. All translated audio and logs are saved under `data/translation_outputs/`.

## ⚙️ Configuration

### Training Parameters
- **Model**: OpenAI Whisper Large v3
- **Languages**: Hindi (hi) and English (en)
- **Batch Size**: 2 (with gradient accumulation)
- **Learning Rate**: 1e-5
- **Epochs**: 10

### Audio Processing
- **Sample Rate**: 16kHz
- **Features**: 80 log-mel spectrograms, 13 MFCC
- **Preprocessing**: Noise reduction, silence trimming, peak normalization

## 🔧 Troubleshooting

### Common Issues

1. **"resample() takes 1 positional argument but 3 were given"**
   - Solution: Update librosa to >= 0.10.0

2. **CUDA out of memory**
   - Solution: Reduce batch size or use CPU mode

3. **Missing audio files**
   - Solution: Check file paths and ensure audio files exist

4. **Model loading errors**
   - Solution: Ensure internet connection for model download

### Performance Tips

- Use GPU for training (significantly faster)
- Increase `n_jobs` for parallel preprocessing
- Monitor disk space during preprocessing (creates many .npy files)

## 📊 Results

The fine-tuned Whisper model achieves competitive performance on Hindi and English ASR tasks. Evaluation metrics include:

- **WER (Word Error Rate)**: Percentage of incorrect words
- **CER (Character Error Rate)**: Percentage of incorrect characters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI Whisper team for the base model
- Hugging Face for the transformers library
- The open-source community for various audio processing tools

