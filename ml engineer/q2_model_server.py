"""
q2_model_server.py

FastAPI wrapper around a trained sklearn model.
This is how I'd serve an ML model when the main backend is Node.js --
keep Python doing what it's good at (ML), expose a simple HTTP endpoint,
and let Node call it like any other service.

To run locally:
    pip install fastapi uvicorn scikit-learn joblib numpy
    python q2_model_server.py

Then hit it with curl or from Node.js:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"features": [150, 1, 1, 1, 2]}'

For the demo I'm training a tiny dummy model on startup so you don't need
a pre-saved model.pkl to run this. In production swap that block for:
    model = joblib.load("model.pkl")
"""

import time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional


# -- demo model ----------------------------------------------------------------
# Remove this in prod and load your real model instead.

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_demo_model():
    """
    Synthetic lead scoring model.
    Features: [company_size, has_budget, has_phone, has_use_case, urgency_score]
    Labels:   0=COLD, 1=WARM, 2=HOT
    Mirrors the scoring rules from q1_lead_classifier.py.
    """
    np.random.seed(42)
    n = 500
    X = np.column_stack([
        np.random.randint(1, 500, n),   # company_size
        np.random.randint(0, 2, n),     # has_budget
        np.random.randint(0, 2, n),     # has_phone
        np.random.randint(0, 2, n),     # has_use_case
        np.random.randint(0, 4, n),     # urgency_score (0-3)
    ])
    scores = (X[:, 0] / 100 + X[:, 1] * 3 + X[:, 2] * 2 + X[:, 3] * 1 + X[:, 4] * 1.5)
    y = np.where(scores >= 7, 2, np.where(scores >= 3, 1, 0))

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=50, random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline

print("Training demo model...")
model = train_demo_model()
LABELS = {0: "COLD", 1: "WARM", 2: "HOT"}
print("Ready.")


# -- app -----------------------------------------------------------------------

app = FastAPI(
    title="KeaBuilder ML Service",
    description="Lead scoring inference endpoint.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in prod
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# -- schemas -------------------------------------------------------------------

class PredictRequest(BaseModel):
    # [company_size, has_budget, has_phone, has_use_case, urgency_score]
    features: List[float]
    request_id: Optional[str] = None  # pass through for tracing / logging

class PredictResponse(BaseModel):
    label: str            # HOT / WARM / COLD
    label_index: int      # 2 / 1 / 0
    probabilities: dict   # {"COLD": 0.1, "WARM": 0.3, "HOT": 0.6}
    confidence: float     # max probability
    latency_ms: float
    request_id: Optional[str]


# -- endpoints -----------------------------------------------------------------

@app.get("/health")
def health():
    """
    Node.js backend polls this every 30s.
    If it goes unhealthy, circuit breaker stops routing requests here.
    """
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Main inference endpoint.

    Node.js calls it like:
        const res = await fetch('http://ml-service:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: [150, 1, 1, 1, 2], request_id: 'req_abc' })
        });
        const { label, confidence } = await res.json();
    """
    if len(req.features) != 5:
        raise HTTPException(
            status_code=422,
            detail=f"Need 5 features, got {len(req.features)}. "
                   f"Order: [company_size, has_budget, has_phone, has_use_case, urgency_score]"
        )

    t0 = time.time()
    X = np.array(req.features).reshape(1, -1)
    idx = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0].tolist()
    latency = round((time.time() - t0) * 1000, 2)

    return PredictResponse(
        label=LABELS[idx],
        label_index=idx,
        probabilities={LABELS[i]: round(p, 4) for i, p in enumerate(probs)},
        confidence=round(max(probs), 4),
        latency_ms=latency,
        request_id=req.request_id
    )


@app.post("/predict/batch")
def predict_batch(requests: List[PredictRequest]):
    """
    Batch inference -- for nightly enrichment runs, bulk re-scoring, etc.
    Doesn't need real-time latency so we can use cheaper infrastructure.
    """
    results = []
    for req in requests:
        try:
            results.append(predict(req))
        except HTTPException as e:
            results.append({"error": e.detail, "request_id": req.request_id})
    return results


# -- run -----------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("q2_model_server:app", host="0.0.0.0", port=8000, reload=False)
