import os
import re
import joblib
import uvicorn
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Path configurations
from config.paths_config import MODEL_OUTPUT_PATH

VECTORIZER_PATH = "artifact/processed/vectorizer.pkl"

# Global references
model = None
vectorizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer
    try:
        if os.path.exists(MODEL_OUTPUT_PATH) and os.path.exists(VECTORIZER_PATH):
            model = joblib.load(MODEL_OUTPUT_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            
            # --- FIX FOR SCIKIT-LEARN VERSION MISMATCH ---
            if not hasattr(model, 'multi_class'):
                model.multi_class = 'auto'
            # ----------------------------------------------

            print("Successfully loaded Model and Vectorizer.")
        else:
            print("Warning: Model or Vectorizer path does not exist.")
    except Exception as e:
        print(f"Critical error loading backend artifacts: {e}")
    yield

app = FastAPI(title="Fake News Detect Application", lifespan=lifespan)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(CURRENT_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

def clean_text(text: str) -> str:
    text = text.lower()
    # Retain alphanumeric characters and spaces only
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    try:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"prediction": None, "alert_class": None}
        )
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Template Loading Error</h1><p>Details: {e}</p>",
            status_code=500
        )

@app.post("/predict", response_class=HTMLResponse)
async def predict_news(request: Request, Input_data: str = Form(...)):
    global model, vectorizer
    
    if model is None or vectorizer is None:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": "Error: Model or Vectorizer is not loaded on server.",
                "alert_class": "warning"
            }
        )
    
    try:
        clean_raw_data = clean_text(Input_data)
        input_feature = vectorizer.transform([clean_raw_data])
        
        raw_pred = model.predict(input_feature)[0]
        
        # Confidence score calculation
        prob_str = ""
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_feature)[0]
            confidence = np.max(probabilities) * 100
            prob_str = f" (Confidence: {confidence:.2f}%)"
            
        # Label handling (Handles 0 as FAKE or string 'FAKE' automatically)
        is_fake = str(raw_pred).upper() in ["0", "FAKE"]

        if is_fake:
            result = f"Fake News Detected!{prob_str}"
            alert_class = "danger"
        else:
            result = f"Real News Confirmed!{prob_str}"
            alert_class = "success"

    except Exception as e:
        result = f"Prediction failed: {str(e)}"
        alert_class = "warning"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": result,
            "alert_class": alert_class,
            "input_text": Input_data
        }
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)