"""Evaluation script for Bangla headline generation model."""

import os
os.environ["USE_TF"] = "0"  # Disable TensorFlow to avoid import errors

import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from app.config import MODEL_DIR, REPORTS_DIR, DATASET_DIR, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
from app.utils import save_json


def load_eval_data(val_path=None, num_samples=500):
    """Load validation data for evaluation."""
    if val_path is None:
        val_path = os.path.join(DATASET_DIR, "processed", "val.json")

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found at {val_path}. Run training first.")

    with open(val_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use a subset for evaluation
    return data[:num_samples]


def generate_predictions(model, tokenizer, data, device, batch_size=8):
    """Generate predictions for evaluation data."""
    predictions = []
    references = []

    model.eval()

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        inputs_text = [f"headline: {item['content']}" for item in batch]
        refs = [item["title"] for item in batch]

        # Tokenize
        inputs = tokenizer(
            inputs_text,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        # Decode
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
        references.extend(refs)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + batch_size}/{len(data)} samples...")

    return predictions, references


def calculate_rouge(predictions, references):
    """Calculate ROUGE scores with custom Bangla tokenizer."""

    class BanglaTokenizer:
        """Custom tokenizer for Bangla text that splits on whitespace."""
        def tokenize(self, text):
            return text.split()

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        tokenizer=BanglaTokenizer()
    )

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": {
            "mean": float(np.mean(rouge1_scores)),
            "std": float(np.std(rouge1_scores)),
            "min": float(np.min(rouge1_scores)),
            "max": float(np.max(rouge1_scores)),
        },
        "rouge2": {
            "mean": float(np.mean(rouge2_scores)),
            "std": float(np.std(rouge2_scores)),
            "min": float(np.min(rouge2_scores)),
            "max": float(np.max(rouge2_scores)),
        },
        "rougeL": {
            "mean": float(np.mean(rougeL_scores)),
            "std": float(np.std(rougeL_scores)),
            "min": float(np.min(rougeL_scores)),
            "max": float(np.max(rougeL_scores)),
        },
    }


def calculate_bleu(predictions, references):
    """Calculate BLEU scores using whitespace tokenization for Bangla."""
    smoothie = SmoothingFunction().method1
    bleu_scores = []

    for pred, ref in zip(predictions, references):
        # Use whitespace tokenization for Bangla
        pred_tokens = pred.split()
        ref_tokens = [ref.split()]

        if not pred_tokens or not ref_tokens[0]:
            bleu_scores.append(0.0)
            continue

        try:
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
            bleu_scores.append(score)
        except Exception:
            bleu_scores.append(0.0)

    return {
        "bleu": {
            "mean": float(np.mean(bleu_scores)),
            "std": float(np.std(bleu_scores)),
            "min": float(np.min(bleu_scores)),
            "max": float(np.max(bleu_scores)),
        }
    }


def generate_pdf_report(evaluation_results, save_path):
    """Generate a PDF evaluation report."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Bangla Headline Generation - Evaluation Report", ln=True, align="C")
    pdf.ln(10)

    # Metadata
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Date: {evaluation_results['timestamp']}", ln=True)
    pdf.cell(0, 8, f"Model: {evaluation_results['model_name']}", ln=True)
    pdf.cell(0, 8, f"Evaluation Samples: {evaluation_results['num_samples']}", ln=True)
    pdf.cell(0, 8, f"Device: {evaluation_results['device']}", ln=True)
    pdf.ln(10)

    # ROUGE Scores
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "ROUGE Scores", ln=True)
    pdf.set_font("Arial", "", 11)

    rouge = evaluation_results["rouge_scores"]
    for metric in ["rouge1", "rouge2", "rougeL"]:
        scores = rouge[metric]
        pdf.cell(0, 8, f"  {metric}: Mean={scores['mean']:.4f}, Std={scores['std']:.4f}", ln=True)

    pdf.ln(5)

    # BLEU Score
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "BLEU Score", ln=True)
    pdf.set_font("Arial", "", 11)

    bleu = evaluation_results["bleu_scores"]["bleu"]
    pdf.cell(0, 8, f"  BLEU: Mean={bleu['mean']:.4f}, Std={bleu['std']:.4f}", ln=True)
    pdf.ln(10)

    # Sample Predictions note (Bangla text can't be rendered in default PDF fonts)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Sample Predictions", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.cell(0, 6, "  (Bangla samples available in evaluation_report.json)", ln=True)

    # Save PDF
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pdf.output(save_path)
    print(f"PDF report saved to: {save_path}")


def evaluate(num_samples=500):
    """Run full evaluation pipeline."""
    print("=" * 60)
    print("Bangla Headline Generation - Evaluation")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.to(device)

    # Load evaluation data
    print("\n[2/4] Loading evaluation data...")
    eval_data = load_eval_data(num_samples=num_samples)
    print(f"Evaluation samples: {len(eval_data)}")

    # Generate predictions
    print("\n[3/4] Generating predictions...")
    predictions, references = generate_predictions(model, tokenizer, eval_data, device)

    # Calculate metrics
    print("\n[4/4] Calculating metrics...")
    rouge_scores = calculate_rouge(predictions, references)
    bleu_scores = calculate_bleu(predictions, references)

    # Compile results
    evaluation_results = {
        "model_name": "google/mt5-small (fine-tuned)",
        "num_samples": len(eval_data),
        "device": str(device),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rouge_scores": rouge_scores,
        "bleu_scores": bleu_scores,
        "samples": [
            {"prediction": pred, "reference": ref}
            for pred, ref in zip(predictions[:10], references[:10])
        ],
    }

    # Save JSON report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    json_report_path = os.path.join(REPORTS_DIR, "evaluation_report.json")
    save_json(evaluation_results, json_report_path)
    print(f"\nJSON report saved to: {json_report_path}")

    # Generate PDF report
    pdf_report_path = os.path.join(REPORTS_DIR, "evaluation_report.pdf")
    generate_pdf_report(evaluation_results, pdf_report_path)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"ROUGE-1: {rouge_scores['rouge1']['mean']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']['mean']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']['mean']:.4f}")
    print(f"BLEU:    {bleu_scores['bleu']['mean']:.4f}")
    print("=" * 60)

    return evaluation_results


if __name__ == "__main__":
    evaluate()
