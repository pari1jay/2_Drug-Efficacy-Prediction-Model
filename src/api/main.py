"""
FastAPI application for drug efficacy prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import DrugEfficacyPredictor
from api.models import DrugProperties, PredictionResponse, HealthResponse

# Initialize FastAPI app
app = FastAPI(
    title="Drug Efficacy Prediction API",
    description="ML-powered API for predicting drug efficacy from molecular properties",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global predictor
    try:
        predictor = DrugEfficacyPredictor()
        print("API startup: Model loaded successfully!")
    except Exception as e:
        print(f"API startup error: {e}")
        predictor = None

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        message="Drug Efficacy Prediction API is running",
        model_loaded=predictor is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        message="Model loaded and ready" if predictor is not None else "Model not loaded",
        model_loaded=predictor is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_efficacy(drug_properties: DrugProperties):
    """Predict drug efficacy from molecular properties"""
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert Pydantic model to dict
        properties_dict = drug_properties.dict()
        
        # Make prediction
        result = predictor.predict_single_drug(properties_dict)
        
        # Return structured response
        return PredictionResponse(
            efficacy_score=result['efficacy_score'],
            confidence=result['confidence'],
            interpretation=result['interpretation'],
            input_properties=properties_dict
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model/info")
async def model_info():
    """Get information about the trained model"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Random Forest Regressor",
        "features": predictor.features,
        "feature_count": len(predictor.features),
        "most_important_feature": "logp",
        "model_accuracy": "83.3% R²",
        "description": "Trained on 1000 synthetic drug molecules with realistic molecular properties"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)