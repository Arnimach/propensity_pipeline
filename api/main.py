import os
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from api.schemas import CustomerFeaturesInput, PredictionResponse


# Point to local MLflow tracking server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Global model cache variable
MODEL_ALIAS = "Production"
MODEL_NAME = "PropensityModel"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the MLflow model onto app.state
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    try:
        app.state.model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully!")
    except Exception as e:
        app.state.model = None
        print(f"Warning: Startup model loading failed: {e}")

    yield  # Serve incoming API traffic

    # Shutdown logic (clean up connections if needed)
    app.state.model = None


app = FastAPI(title="Customer Propensity Real-Time Inference API", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": getattr(app.state, "model", None) is not None,
        "model_alias": MODEL_ALIAS
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_propensity(features: CustomerFeaturesInput, request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Production model not loaded.")

    # Convert incoming Pydantic payload to single-row DataFrame
    input_df = pd.DataFrame([features.model_dump()])

    try:
        # Run inference using pyfunc predict method
        preds = model.predict(input_df)
        pred_value = int(preds[0])

        return PredictionResponse(
            prediction=pred_value,
            high_value_flag=(pred_value == 1),
            model_version_used=MODEL_ALIAS
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
