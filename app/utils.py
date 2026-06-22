"""Utility functions for the application."""

import os
import json
import torch
from datetime import datetime


def get_device():
    """Get the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(data, filepath):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath):
    """Load data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_info(model_dir):
    """Get model information from saved model directory."""
    info = {
        "model_name": "google/mt5-small",
        "task": "Text-to-Text Generation (Bangla Headline)",
        "framework": "PyTorch + HuggingFace Transformers",
        "language": "Bangla (bn)",
    }

    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        config = load_json(config_path)
        info["vocab_size"] = config.get("vocab_size", "N/A")
        info["model_type"] = config.get("model_type", "N/A")
        info["d_model"] = config.get("d_model", "N/A")
        info["num_heads"] = config.get("num_heads", "N/A")
        info["num_layers"] = config.get("num_layers", "N/A")

    return info


def get_training_metrics(reports_dir):
    """Load training metrics from reports directory."""
    metrics_path = os.path.join(reports_dir, "training_metrics.json")
    if os.path.exists(metrics_path):
        return load_json(metrics_path)
    return None


def format_timestamp():
    """Get current timestamp as formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
