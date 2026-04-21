from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from typing import Optional
from src.preprocessing.text_cleaner import TextCleaner
from src.features.tfidf_extractor import TfidfExtractor
from src.models.baseline import BaselineModel
from src.inference.explainer import ModelExplainer
from api.database import log_prediction
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="API for detecting fake news using ML models.",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
cleaner = None
extractor = None
model = None
explainer = None

# Schemas
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float
    explanation: Optional[str] = None

@app.on_event("startup")
async def load_models():
    """Load the model and vectorizer on startup."""
    global cleaner, extractor, model, explainer
    try:
        print("Loading models...")
        cleaner = TextCleaner()
        extractor = TfidfExtractor()
        extractor.load("models/tfidf_vectorizer.pkl")
        
        model = BaselineModel(model_type='logistic_regression')
        model.load("models/baseline_model.pkl")
        
        explainer = ModelExplainer(model, extractor)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load models. {e}")
        # Not throwing exception here so health check can still run

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Endpoint to predict if news is real or fake."""
    if not model or not extractor or not model.is_fitted:
        raise HTTPException(status_code=503, detail="Models are not loaded or fitted yet.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    try:
        # Preprocess
        cleaned_text = cleaner.clean_text(request.text)
        features = extractor.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Explain
        explanation = explainer.explain_prediction(request.text, cleaned_text)
        
        # Map label (0 = Fake, 1 = Real based on data loader)
        label_str = "REAL" if prediction == 1 else "FAKE"
        confidence = float(probabilities[prediction])
        
        # Log to database
        log_prediction(request.text, label_str, confidence, explanation)
        
        return PredictResponse(
            label=label_str,
            confidence=confidence,
            explanation=explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
