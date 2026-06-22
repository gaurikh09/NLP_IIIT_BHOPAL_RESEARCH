"""Training script for Bangla headline generation using mT5-small."""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

from app.config import (
    MODEL_NAME,
    MODEL_DIR,
    REPORTS_DIR,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    EVAL_STEPS,
)
from training.preprocess import prepare_dataset


def tokenize_function(examples, tokenizer):
    """Tokenize input articles and target headlines."""
    inputs = [f"headline: {content}" for content in examples["content"]]
    targets = examples["title"]

    # Tokenize inputs with padding to max_length
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )

    # Tokenize targets WITHOUT padding (let DataCollator handle it)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer):
    """Compute ROUGE metrics during evaluation."""
    from rouge_score import rouge_scorer

    predictions, labels = eval_pred

    # Replace -100 in predictions with pad token id (can't decode negative values)
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in labels with pad token id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, label in zip(decoded_preds, decoded_labels):
        if not pred:
            pred = " "
        if not label:
            label = " "
        scores = scorer.score(label, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores),
    }


def plot_loss_curve(trainer, save_path):
    """Plot and save the training loss curve."""
    logs = trainer.state.log_history

    train_losses = [log["loss"] for log in logs if "loss" in log]
    steps = [log["step"] for log in logs if "loss" in log]

    if not train_losses:
        print("No training loss data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, "b-", label="Training Loss", linewidth=2)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Curve - Bangla Headline Generation (mT5-small)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to: {save_path}")


def train():
    """Main training function."""
    print("=" * 60)
    print("Bangla News Headline Generation - Training")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("=" * 60)

    # Prepare dataset
    print("\n[1/6] Preparing dataset...")
    train_data, val_data = prepare_dataset()

    # Load tokenizer and model
    print("\n[2/6] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")

    # Create HuggingFace datasets
    print("\n[3/6] Tokenizing data...")
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["title", "content"],
        desc="Tokenizing train data",
    )

    val_dataset = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["title", "content"],
        desc="Tokenizing validation data",
    )

    # Use subset for faster evaluation during training (full eval after training)
    eval_dataset = val_dataset.select(range(min(1000, len(val_dataset))))

    # Verify tokenization
    sample = train_dataset[0]
    print(f"Sample input_ids length: {len(sample['input_ids'])}")
    print(f"Sample labels length: {len(sample['labels'])}")
    print(f"Sample labels (first 10): {sample['labels'][:10]}")
    non_special = [t for t in sample['labels'] if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
    print(f"Non-special tokens in labels: {len(non_special)}")

    # Data collator - pads labels and replaces pad_token_id with -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # Training arguments
    print("\n[4/6] Setting up training arguments...")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,  # Can use larger batch for eval (no gradients)
        gradient_accumulation_steps=2,  # Effective batch size = 4
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        logging_steps=100,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        fp16=False,  # mT5 has issues with fp16 causing nan gradients
        bf16=False,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        report_to="none",
        optim="adamw_torch",
        warmup_steps=500,
        save_total_limit=2,
    )

    # Trainer
    print("\n[5/6] Starting training...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    # Train
    start_time = datetime.now()
    train_result = trainer.train()
    end_time = datetime.now()
    training_duration = str(end_time - start_time)

    # Save model
    print("\n[6/6] Saving model...")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to: {MODEL_DIR}")

    # Save training metrics
    metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        "total_steps": train_result.global_step,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "training_duration": training_duration,
        "device": str(device),
        "model_name": MODEL_NAME,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Add eval metrics if available
    eval_results = trainer.evaluate()
    metrics["eval_loss"] = eval_results.get("eval_loss", None)
    metrics["eval_rouge1"] = eval_results.get("eval_rouge1", None)
    metrics["eval_rouge2"] = eval_results.get("eval_rouge2", None)
    metrics["eval_rougeL"] = eval_results.get("eval_rougeL", None)

    metrics_path = os.path.join(REPORTS_DIR, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Training metrics saved to: {metrics_path}")

    # Plot loss curve
    loss_curve_path = os.path.join(REPORTS_DIR, "loss_curve.png")
    plot_loss_curve(trainer, loss_curve_path)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Duration: {training_duration}")
    print(f"Final Train Loss: {train_result.training_loss:.4f}")
    if metrics.get("eval_rouge1"):
        print(f"Eval ROUGE-1: {metrics['eval_rouge1']:.4f}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    train()
