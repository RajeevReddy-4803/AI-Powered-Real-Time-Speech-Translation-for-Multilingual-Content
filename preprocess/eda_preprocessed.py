import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display


def ensure_out_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_duration_distributions(df: pd.DataFrame, out_dir: Path):
    durations = df['duration_sec'].astype(float).clip(lower=0.0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.histplot(durations, bins=50, ax=axes[0], kde=True, color="#4C78A8")
    axes[0].set_title("Duration Histogram (sec)")
    sns.boxplot(x=durations, ax=axes[1], color="#72B7B2")
    axes[1].set_title("Duration Boxplot")
    sns.violinplot(x=durations, ax=axes[2], color="#F58518")
    axes[2].set_title("Duration Violin")
    plt.tight_layout()
    fig.savefig(out_dir / "duration_distributions.png", dpi=200)
    plt.close(fig)

    # Save basic stats
    stats = durations.describe(percentiles=[0.01, 0.05, 0.1, 0.9, 0.95, 0.99])
    stats.to_csv(out_dir / "duration_stats.csv", header=["value"])


def plot_random_spectrograms(df: pd.DataFrame, out_dir: Path, num_examples: int = 6, feature_col: str = 'logmel_npy'):
    df_feat = df[df[feature_col].notna() & df[feature_col].astype(str).str.len() > 0]
    if df_feat.empty:
        return
    examples = df_feat.sample(n=min(num_examples, len(df_feat)), random_state=42)
    ncols = 3
    nrows = int(np.ceil(len(examples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3))
    axes = np.array(axes).reshape(nrows, ncols)
    for (idx, row), ax in zip(examples.iterrows(), axes.flatten()):
        try:
            feat = np.load(row[feature_col])
            librosa.display.specshow(feat, x_axis='time', y_axis='mel', sr=int(row.get('sr', 16000)), ax=ax)
            ax.set_title(Path(row['out_wav']).name)
        except Exception:
            ax.set_visible(False)
    # Hide any extra axes
    total_axes = nrows * ncols
    for k in range(len(examples), total_axes):
        axes.flatten()[k].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "random_logmel_examples.png", dpi=200)
    plt.close(fig)


def main(manifest_csv: str, out_dir: str, num_examples: int = 6):
    out_path = Path(out_dir)
    ensure_out_dir(out_path)
    df = pd.read_csv(manifest_csv)
    if 'duration_sec' in df.columns:
        plot_duration_distributions(df, out_path)
    plot_random_spectrograms(df, out_path, num_examples=num_examples, feature_col='logmel_npy')

def run_eda(manifest_csv):
    df = pd.read_csv(manifest_csv)
    print(f"Loaded {len(df)} samples from {manifest_csv}\n")

    if "duration_sec" in df.columns:
        print(f"⏱️ Duration stats (sec):")
        print(df["duration_sec"].describe())
        plt.hist(df["duration_sec"], bins=50)
        plt.title("Clip Duration Distribution (seconds)")
        plt.xlabel("Duration (s)")
        plt.ylabel("Count")
        plt.show()

    if "transcript" in df.columns:
        df["char_len"] = df["transcript"].astype(str).apply(len)
        print(f"\n📝 Transcript length stats (characters):")
        print(df["char_len"].describe())
        plt.hist(df["char_len"], bins=50)
        plt.title("Transcript Length Distribution")
        plt.xlabel("Characters")
        plt.ylabel("Count")
        plt.show()

    if "lang" in df.columns:
        print("\n🌐 Language distribution:")
        print(df["lang"].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to preprocess_manifest.csv")
    parser.add_argument("--out_dir", required=True, help="Directory to save EDA outputs")
    parser.add_argument("--num_examples", type=int, default=6)
    args = parser.parse_args()
    run_eda(args.manifest)
    main(args.manifest, args.out_dir, args.num_examples)


