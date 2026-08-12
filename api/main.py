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
    """
    Manages the application lifecycle by loading the production ML model at startup
    and releasing resources upon shutdown.

    During startup, fetches the latest model artifact tagged with `MODEL_ALIAS` from 
    the MLflow Model Registry and attaches it to `app.state.model`. On shutdown, resets 
    the state reference to ensure clean memory release.

    Args:
        app (FastAPI): The main FastAPI application instance.

    Yields:
        None: Transfers execution control to FastAPI to begin serving HTTP endpoints.
    """
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
    """
    Performs an operational health check of the API and its backend dependencies.

    Inspects whether the FastAPI application state contains an initialized MLflow 
    model instance and returns runtime metadata.

    Returns:
        dict: A dictionary containing:
            - **status** (*str*): Liveness state indicator ('healthy').
            - **model_loaded** (*bool*): `True` if the ML model is successfully 
            loaded in state, `False` otherwise.
            - **model_alias** (*str*): The active MLflow model registry 
            alias (e.g., 'Production').
    """
    return {
        "status": "healthy",
        "model_loaded": getattr(app.state, "model", None) is not None,
        "model_alias": MODEL_ALIAS
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_propensity(features: CustomerFeaturesInput, request: Request):
    """
    Generates real-time propensity scores for incoming web customer feature payloads.

    Retrieves the pre-loaded MLflow `pyfunc` model from the application state, converts 
    the validated Pydantic schema into a single-row Pandas DataFrame, and executes model 
    inference.

    Args:
        features (CustomerFeaturesInput): Validated request body containing session 
            and demographic characteristics.
        request (Request): Raw HTTP request instance used to access `app.state`.

    Returns:
        PredictionResponse: Standardized response object containing the binary class prediction, 
        a high-value customer boolean flag, and the model version alias used.

    Raises:
        HTTPException:
            - **503 Service Unavailable**: Raised if the ML model is not loaded in `app.state`.
            - **500 Internal Server Error**: Raised if an exception occurs during pipeline inference.
    """

    
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
