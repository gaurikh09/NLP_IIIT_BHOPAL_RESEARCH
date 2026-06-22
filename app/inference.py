"""Inference engine for Bangla headline generation."""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.config import MODEL_DIR, MODEL_NAME, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, BEAM_NUM
from app.utils import get_device


class HeadlineGenerator:
    """Bangla headline generation inference engine."""

    def __init__(self, model_path=None):
        """Initialize the generator with a trained model."""
        self.device = get_device()
        self.model_path = model_path or MODEL_DIR
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the trained model and tokenizer."""
        if os.path.exists(self.model_path):
            print(f"Loading model from: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        else:
            print(f"No saved model found at {self.model_path}. Loading base model: {MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")

    def generate(self, article_text, max_length=MAX_TARGET_LENGTH, num_beams=BEAM_NUM):
        """Generate a headline from the given article text.

        Args:
            article_text: Full Bangla article text.
            max_length: Maximum length of generated headline.
            num_beams: Number of beams for beam search.

        Returns:
            dict with generated headline and metadata.
        """
        # Prepare input
        input_text = f"headline: {article_text}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
            )

        # Decode
        generated_headline = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "headline": generated_headline,
            "input_length": len(article_text),
            "device": str(self.device),
        }

    def is_loaded(self):
        """Check if model is loaded successfully."""
        return self.model is not None and self.tokenizer is not None
