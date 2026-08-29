import os
import re
import joblib
import uvicorn
import numpy as np
from prometheus_client import generate_latest, Counter, Gauge, CONTENT_TYPE_LATEST # इम्पोर्ट्स को सुधारा
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request, Response # 'Response' यहाँ इम्पोर्ट किया
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ALIBI DETECT DEPENDENCIES
from alibi_detect.cd import KSDrift

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

# File Paths Alignment
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "artifacts", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "artifact", "processed", "vectorizer.pkl")
BASELINE_DATA_PATH = os.path.join(BASE_DIR, "artifacts", "baseline_features.npy")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv") # Fallback raw training path



# Prometheus Counters
prediction_count = Counter('prediction_count', "Number of prediction count")
drift_count = Counter('drift_count', "Number of times data drift is detected")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

@asynccontextmanager
async def lifespan(app: FastAPI):
   
    try:
        if os.path.exists(MODEL_OUTPUT_PATH) and os.path.exists(VECTORIZER_PATH):
            model = joblib.load(MODEL_OUTPUT_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            
            if not hasattr(model, 'multi_class'):
                model.multi_class = 'auto'

            print("Successfully loaded Model and Vectorizer.")
            
            # Baseline Extraction Strategy
            if os.path.exists(BASELINE_DATA_PATH):
                X_baseline = np.load(BASELINE_DATA_PATH)
                print("Loaded Baseline features from baseline_features.npy")
            elif os.path.exists(RAW_DATA_PATH):
                print("Generating baseline reference matrix from training dataset...")
                df = pd.read_csv(RAW_DATA_PATH)
                df.fillna(" ", inplace=True)
                content = df["title"].astype(str) + " " + df["text"].astype(str)
                cleaned = [clean_text(t) for t in content.iloc[:500]]
                X_baseline = vectorizer.transform(cleaned).toarray()
                os.makedirs(os.path.dirname(BASELINE_DATA_PATH), exist_ok=True)
                np.save(BASELINE_DATA_PATH, X_baseline)
            else:
                X_baseline = None

            if X_baseline is not None:
                if hasattr(X_baseline, "toarray"):
                    X_baseline = X_baseline.toarray()
                    
                # Setup KSDrift with appropriate sensitivity p_val threshold
                drift_detector = KSDrift(x_ref=X_baseline, p_val=0.001)
                print("KSDrift Initialized with Reference Features Matrix Shape:", X_baseline.shape)

    except Exception as e:
        print(f"Error loading backend artifacts: {e}")
    yield

app = FastAPI(title="Fake News Detection Pipeline", lifespan=lifespan)

TEMPLATES_DIR = os.path.join(CURRENT_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": None, "alert_class": None, "input_text": ""}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict_news(request: Request, Input_data: str = Form(...)):
   
    if model is None or vectorizer is None:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": "Error: Server artifacts not loaded.",
                "alert_class": "warning",
                "input_text": Input_data
            }
        )
    
    try:
        # हर प्रेडिक्शन रिक्वेस्ट पर प्रेडिक्शन काउंटर बढ़ाएं
        prediction_count.inc()
        
        clean_raw_data = clean_text(Input_data)
        input_feature = vectorizer.transform([clean_raw_data])
        
        # Convert matrix to check non-zero vocabulary feature matches
        x_query = input_feature.toarray() if hasattr(input_feature, "toarray") else np.array(input_feature)
        vocab_matches = np.count_nonzero(x_query)
        total_words = len(clean_raw_data.split())
        
        # 1. DRIFT DETECTION RULE
        is_drift = 0
        if total_words > 3 and vocab_matches == 0:
            is_drift = 1

        # 2. MUTUALLY EXCLUSIVE ROUTING
        if is_drift == 1:
            # ड्रिफ्ट होने पर ड्रिफ्ट काउंटर बढ़ाएं
            drift_count.inc()
            result = "Data Drift Detected! (Input Out-Of-Distribution)"
            alert_class = "warning"
        else:
            # Predict Label
            raw_pred = model.predict(input_feature)[0]
            
            prob_str = ""
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_feature)[0]
                confidence = np.max(probabilities) * 100
                prob_str = f" (Confidence: {confidence:.2f}%)"
                
            # String / Integer Label Handling for Logistic Regression & Naive Bayes
            pred_val = str(raw_pred).strip().upper()
            is_fake = pred_val in ["0", "FAKE", "LABEL_0"]

            if is_fake:
                result = f"Fake News Detected!{prob_str}"
                alert_class = "danger"
            else:
                result = f"Real News Confirmed!{prob_str}"
                alert_class = "success"

    except Exception as e:
        result = f"Pipeline execution error: {str(e)}"
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


@app.get("/metrics")
def metrics():
    # प्रॉमिसियस के लिए सही कंटेंट टाइप (text/plain) के साथ रिस्पॉन्स देना
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
