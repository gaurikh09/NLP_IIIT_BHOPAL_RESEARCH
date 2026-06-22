"""Data preprocessing for Bangla headline generation training."""

import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    DATASET_DIR,
    MIN_CONTENT_LENGTH,
    TOTAL_SAMPLES,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
)


def load_dataset(dataset_path=None):
    """Load the Bangla Newspaper Dataset from JSON file.

    Args:
        dataset_path: Path to data.json file.

    Returns:
        List of article records with 'title' and 'content' fields.
    """
    if dataset_path is None:
        dataset_path = os.path.join(DATASET_DIR, "data.json")

    print(f"Loading dataset from: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Please place your data.json file in the dataset/ directory."
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total records loaded: {len(data)}")
    return data


def clean_data(data):
    """Clean and filter the dataset.

    - Remove articles with content length < 100 characters.
    - Remove records with missing title or content.

    Args:
        data: List of raw article records.

    Returns:
        List of cleaned records.
    """
    cleaned = []
    skipped = 0

    for record in data:
        title = record.get("title", "").strip()
        content = record.get("content", "").strip()

        # Skip if title or content is missing
        if not title or not content:
            skipped += 1
            continue

        # Skip if content is too short
        if len(content) < MIN_CONTENT_LENGTH:
            skipped += 1
            continue

        cleaned.append({"title": title, "content": content})

    print(f"Cleaned records: {len(cleaned)} | Skipped: {skipped}")
    return cleaned


def split_data(data, train_size=TRAIN_SAMPLES, val_size=VAL_SAMPLES):
    """Split data into train and validation sets.

    Args:
        data: Cleaned dataset.
        train_size: Number of training samples.
        val_size: Number of validation samples.

    Returns:
        Tuple of (train_data, val_data).
    """
    total_needed = train_size + val_size

    if len(data) < total_needed:
        print(f"Warning: Only {len(data)} samples available. Adjusting split.")
        train_size = int(len(data) * 0.9)
        val_size = len(data) - train_size

    # Shuffle data
    import random
    random.seed(42)
    random.shuffle(data)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]

    print(f"Train samples: {len(train_data)} | Validation samples: {len(val_data)}")
    return train_data, val_data


def prepare_dataset(dataset_path=None):
    """Full preprocessing pipeline.

    Args:
        dataset_path: Optional path to dataset.

    Returns:
        Tuple of (train_data, val_data).
    """
    # Load
    data = load_dataset(dataset_path)

    # Clean
    cleaned = clean_data(data)

    # Split
    train_data, val_data = split_data(cleaned)

    # Save processed data
    processed_dir = os.path.join(DATASET_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    train_path = os.path.join(processed_dir, "train.json")
    val_path = os.path.join(processed_dir, "val.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"Saved train data to: {train_path}")
    print(f"Saved validation data to: {val_path}")

    # Dataset statistics
    stats = {
        "total_raw_records": len(data),
        "total_cleaned_records": len(cleaned),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "min_content_length_filter": MIN_CONTENT_LENGTH,
        "avg_content_length": sum(len(r["content"]) for r in cleaned) / max(len(cleaned), 1),
        "avg_title_length": sum(len(r["title"]) for r in cleaned) / max(len(cleaned), 1),
    }

    stats_path = os.path.join(processed_dir, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Dataset statistics saved to: {stats_path}")
    return train_data, val_data


if __name__ == "__main__":
    prepare_dataset()
