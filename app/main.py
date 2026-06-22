"""FastAPI application for Bangla Headline Generation."""

import os
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional

from app.config import MODEL_DIR, REPORTS_DIR, BASE_DIR
from app.utils import get_model_info, get_training_metrics, format_timestamp, save_json, load_json
from app.inference import HeadlineGenerator

# Initialize FastAPI app
app = FastAPI(
    title="Bangla News Headline Generator",
    description="Generate Bangla news headlines from full articles using fine-tuned mT5-small model.",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend directory
frontend_dir = os.path.join(BASE_DIR, "frontend")

# Initialize model (lazy loading)
generator = None
predictions_history = []

PREDICTIONS_FILE = os.path.join(REPORTS_DIR, "predictions_history.json")


def get_generator():
    """Lazy load the headline generator."""
    global generator
    if generator is None:
        generator = HeadlineGenerator()
    return generator


def load_predictions():
    """Load prediction history from file."""
    global predictions_history
    if os.path.exists(PREDICTIONS_FILE):
        predictions_history = load_json(PREDICTIONS_FILE)


def save_predictions():
    """Save prediction history to file."""
    save_json(predictions_history, PREDICTIONS_FILE)


# Load predictions on startup
load_predictions()


# Request/Response models
class GenerateRequest(BaseModel):
    article: str = Field(..., min_length=10, description="Full Bangla article text")
    actual_headline: Optional[str] = Field(None, description="Actual headline for comparison")


class GenerateResponse(BaseModel):
    headline: str
    actual_headline: Optional[str] = None
    input_length: int
    device: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend dashboard."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found. Visit /docs for API.</h1>")


@app.get("/style.css")
async def serve_css():
    """Serve CSS file."""
    css_path = os.path.join(frontend_dir, "style.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")


@app.get("/script.js")
async def serve_js():
    """Serve JS file."""
    js_path = os.path.join(frontend_dir, "script.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gen = get_generator()
    return HealthResponse(
        status="running",
        model_loaded=gen.is_loaded(),
        device=str(gen.device),
        timestamp=format_timestamp(),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_headline(request: GenerateRequest):
    """Generate a Bangla headline from an article."""
    try:
        gen = get_generator()

        if not gen.is_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded.")

        result = gen.generate(request.article)

        response = GenerateResponse(
            headline=result["headline"],
            actual_headline=request.actual_headline,
            input_length=result["input_length"],
            device=result["device"],
            timestamp=format_timestamp(),
        )

        # Store prediction
        prediction_record = {
            "article_preview": request.article[:200],
            "generated_headline": result["headline"],
            "actual_headline": request.actual_headline,
            "timestamp": format_timestamp(),
        }
        predictions_history.append(prediction_record)
        save_predictions()

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info():
    """Get model information."""
    info = get_model_info(MODEL_DIR)
    return info


@app.get("/training-metrics")
async def training_metrics():
    """Get training metrics."""
    metrics = get_training_metrics(REPORTS_DIR)
    if metrics is None:
        return {"message": "No training metrics available. Train the model first."}
    return metrics


@app.get("/predictions")
async def get_predictions():
    """Get prediction history."""
    return {"predictions": predictions_history, "total": len(predictions_history)}


@app.delete("/predictions")
async def clear_predictions():
    """Clear prediction history."""
    global predictions_history
    predictions_history = []
    save_predictions()
    return {"message": "Prediction history cleared."}


@app.get("/predictions/export")
async def export_predictions_csv():
    """Export predictions as CSV."""
    import csv
    import io
    from fastapi.responses import StreamingResponse

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Article Preview", "Generated Headline", "Actual Headline"])

    for pred in predictions_history:
        writer.writerow([
            pred.get("timestamp", ""),
            pred.get("article_preview", ""),
            pred.get("generated_headline", ""),
            pred.get("actual_headline", ""),
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
