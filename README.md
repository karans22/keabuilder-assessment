# keabuilder-assessment
# KeaBuilder — Dream Reflection Media Technical Assessment

**Karan Sharma** · [karansharma2202@gmail.com](mailto:karansharma2202@gmail.com)

---

## Overview

This repo contains my submissions for the AI Engineer and ML Engineer assessments for Dream Reflection Media's KeaBuilder platform — a SaaS tool for funnels, lead capture, and marketing automation.

Both assessments share a common context: design and implement AI/ML features that enhance lead handling, content generation, and user experience inside KeaBuilder.

---

## Repository Structure

```
keabuilder-assessment/
├── ai_engineer/
│   └── q1_lead_classifier.py       # Q1: Lead classification + response pipeline
│
├── ml_engineer/
│   ├── q1_similarity_search.py     # Q1: Semantic similarity search engine
│   ├── q2_model_server.py          # Q2: FastAPI ML model server (Node.js compatible)
│   └── q3_schema.sql               # Q3: PostgreSQL schema for inputs + predictions
│
└── README.md
```

Full written answers for all questions (Q2–Q7 for both assessments) are included below.

---

## Part 1 — AI Engineer Assessment

### Q1 · Lead Classification & Intelligent Response System · [`ai_engineer/q1_lead_classifier.py`](ai_engineer/q1_lead_classifier.py)

**How to run:**
```bash
python ai_engineer/q1_lead_classifier.py
```
No API keys or external dependencies needed — runs with standard Python 3.

**What it does:**

Incoming form data goes through a two-stage pipeline. First, a rule-based scorer checks high-signal fields (budget, company size, urgency language, phone number). That handles ~80% of cases correctly and is instant. For the rest, an LLM call with a structured prompt handles nuanced classification. I wrote the prompts so they're ready to drop in — they're in the file as `CLASSIFY_PROMPT` and `RESPOND_PROMPT`.

**Classification signals:**
- Budget mentioned → +3
- Company size > 100 → +3
- Urgency language in message → +3 (ASAP, migrating, end of quarter, etc.)
- Specific feature mentioned → +2 (funnel, chatbot, automation, CRM)
- Phone number provided → +2
- Score ≥ 7 → HOT · 3–6 → WARM · < 3 → COLD

**Making responses feel human:** The key is referencing what the lead *actually wrote* — not generic product copy. The response prompt explicitly instructs the LLM to use their first name and reference their specific use_case or message text. HOT responses are direct and suggest a call this week. WARM responses drop something useful with no pressure. COLD responses are under 3 sentences.

**Incomplete inputs:** Critical missing fields (name, email) block classification and trigger a clarification flow — max 2 questions, conversational tone. Non-critical missing fields (company, use_case) don't block processing; we classify with available data and tag the lead for enrichment.

**Sample output:**
```json
{
  "status": "ok",
  "lead_id": "KB-1AEDB8",
  "score": {
    "classification": "HOT",
    "confidence": 0.91,
    "why": "scored 10 points, signals: mentioned budget, 100+ person company, urgency in message, phone provided",
    "signals_found": ["mentioned budget", "100+ person company", "urgency in message", "phone provided"]
  },
  "response": {
    "subject": "Quick question about GrowthCo + KeaBuilder",
    "body": "Hey Priya,\n\nSaw your note about funnel automation ASAP...",
    "cta": "Pick a time that works for you",
    "follow_up_in_days": 1
  }
}
```

---

### Q2 · Multi-Provider Content Generation Routing

A `ContentRouter` class reads a `content_type` field from the request and dispatches to the right provider adapter. Each adapter normalizes its response so the rest of the platform doesn't care which provider ran.

| content_type | Primary | Fallback |
|---|---|---|
| `image` | DALL-E 3 | Stable Diffusion (Replicate) |
| `video` | RunwayML | Pika Labs |
| `voice` | ElevenLabs | Azure TTS → Edge TTS |

**Request from builder UI:**
```json
POST /api/generate
{
  "content_type": "image",
  "prompt": "professional headshot, warm lighting",
  "user_id": "u_123",
  "funnel_id": "f_456",
  "options": { "style": "photorealistic", "aspect_ratio": "1:1" }
}
```

**Normalized response from any adapter:**
```json
{
  "asset_url": "https://cdn.keabuilder.com/assets/abc123.png",
  "provider": "dalle",
  "generation_id": "gen_789",
  "status": "ready"
}
```

Generated assets are uploaded to S3/Cloudflare R2, a CDN URL is stored in the DB, and they appear in the builder's media library. Re-generating always creates a new asset — nothing gets overwritten.

---

### Q3 · Personalised AI Images with LoRA

LoRA lets us layer fine-tuned weights on a frozen base model (SDXL). Instead of training a full model per user, we train a small adapter on 15–30 of their photos. Training takes ~15 minutes on serverless GPU (Replicate/Modal.com) and costs under $1.

**Pipeline:**
1. User uploads 15–30 reference photos via KeaBuilder UI
2. Backend kicks off a DreamBooth LoRA training job
3. Output `.safetensors` file (~150MB) stored on S3, linked to the user's account as their "Brand Profile"
4. On generation requests, inject `lora_weights` + `lora_scale` alongside the base model call

**Inference request:**
```json
{
  "content_type": "image",
  "prompt": "ohwx person standing in a modern office",
  "lora_weights": "s3://keabuilder/loras/user_u123_v2.safetensors",
  "lora_scale": 0.85,
  "base_model": "sdxl"
}
```

The trigger word (`ohwx person`) is baked in during training — must appear in every prompt or the LoRA weights don't activate. LoRA files are cached per user with a 24h TTL.

---

### Q4 · Face & Text Similarity Search

**Storage:** Text embeddings via sentence-transformers (MiniLM-L6) → stored in PostgreSQL with the pgvector extension. Face/image embeddings via CLIP or DeepFace — same table.

**Retrieval:**
```sql
SELECT id, input_text, 1 - (input_vector <=> $1::vector) AS similarity
FROM user_inputs
WHERE user_id = $2
ORDER BY input_vector <=> $1::vector
LIMIT 10;
```
HNSW index handles this at ~5ms for 100K records.

**Matching:** Cosine similarity > 0.80 → surfaced as "similar" in the UI. Final ranking: `0.7 × vector_sim + 0.3 × recency_score`.

---

### Q5 · Fallback Strategy When AI Services Fail

**Provider chains** — if A fails, B runs automatically, user sees nothing:
```
Image:  DALL-E 3  →  Stable Diffusion  →  placeholder
Text:   GPT-4o    →  Claude 3.5        →  Gemini Pro
Voice:  ElevenLabs → Azure TTS         →  Edge TTS (always free)
```

**Circuit breaker:** 3 failures in 5 minutes → provider marked degraded for 10 minutes before retrying.

**UX:** Skeleton loader + "Generating..." spinner during retries. If all providers fail: job queued in BullMQ + Redis with exponential backoff (2s → 4s → 8s), toast shown, in-app + email notification when done.

---

### Q6 · High-Volume AI Request Handling

**Performance:** HTTP endpoint returns a `job_id` immediately; generation happens in the background. LLM text streamed via SSE — same total latency, feels much faster to the user.

**Cost:** Hash `(prompt + model + params)` → check Redis before calling any API. Seen this cut API spend by 30–50%. Free plan users get Gemini Flash / SD Turbo; paid users get GPT-4o / DALL-E 3. Batch API for non-real-time jobs (50% cost reduction).

**Reliability:** Provider health checks every 30s. Dead letter queue for jobs that exhaust retries. Every generation logged: provider, latency, token count, cost. Alert on P95 > 10s.

---

### Q7 · Tools & Frameworks

- **LLMs:** Gemini 1.5 Flash — built a fully automated YouTube channel ([Anime Vault project](https://github.com/karans22)) with script generation, structured JSON output, TTS, video assembly, and daily GitHub Actions upload pipeline
- **Automation:** GitHub Actions, Playwright (job-auto-apply project), openpyxl
- **Data engineering:** PySpark, Hive, Kafka streaming, dbt with Medallion architecture
- **Cloud:** AWS S3/Lambda concepts, Azure, GCP basics
- **Backend:** Python (FastAPI), Node.js familiarity
- **ML:** LightGBM + SHAP, scikit-learn, Prophet, XGBoost, SARIMA, LSTM
- **Media:** MoviePy, Edge TTS, Pexels API, YouTube Data API v3

---

## Part 2 — ML Engineer Assessment

### Q1 · Semantic Similarity Search · [`ml_engineer/q1_similarity_search.py`](ml_engineer/q1_similarity_search.py)

**How to run:**
```bash
python ml_engineer/q1_similarity_search.py
```
Zero dependencies — pure Python 3.

Implements TF-IDF vectorization + cosine similarity from scratch. No sklearn, no external model. The `LeadMatcher` class indexes a corpus of existing lead texts and finds the closest match to any new query.

I kept it dependency-free on purpose: for a demo that matches lead text to existing leads or prompts, TF-IDF is accurate enough and runs anywhere. In production I'd swap in sentence-transformers + pgvector for semantic (meaning-level) matching — the architecture is identical, just a different vectorizer.

**Sample output:**
```
Query: 'automate follow-up emails and lead management'
  #1 [0.68] lead_001 (HOT) — We need to automate lead capture and follow-up emails...
  #2 [0.08] lead_005 (HOT) — Need CRM integration and email drip campaigns...

Query: 'chatbot for customer support on website'
  #1 [0.60] lead_002 (WARM) — Looking for a chatbot to handle incoming queries...
```

---

### Q2 · Serving an ML Model from Node.js · [`ml_engineer/q2_model_server.py`](ml_engineer/q2_model_server.py)

**How to run:**
```bash
pip install fastapi uvicorn scikit-learn numpy
python ml_engineer/q2_model_server.py
# server starts on http://localhost:8000
```

**Test with curl:**
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [150, 1, 1, 1, 2], "request_id": "test_001"}'
```

**Response:**
```json
{
  "label": "HOT",
  "label_index": 2,
  "probabilities": {"COLD": 0.0009, "WARM": 0.0085, "HOT": 0.9907},
  "confidence": 0.9907,
  "latency_ms": 2.32,
  "request_id": "test_001"
}
```

**Architecture:** Python FastAPI microservice in Docker, Node.js calls it over internal HTTP. Model loads once at startup. Includes `/health` endpoint for Node.js to poll, `/predict` for single inference, and `/predict/batch` for bulk runs.

---

### Q3 · Database Schema · [`ml_engineer/q3_schema.sql`](ml_engineer/q3_schema.sql)

**How to apply:**
```bash
# requires PostgreSQL with pgvector extension
psql -U your_user -d your_db -f ml_engineer/q3_schema.sql
```

Two tables:
- `user_inputs` — raw text, embedding vector, type, metadata
- `ml_predictions` — label, confidence, feature snapshot, latency, model version

The tables are intentionally separate so we can re-run predictions on old inputs when deploying a new model version, without touching the original data.

---

### Q4 · Slow ML Responses in UI

Show a skeleton loader immediately on click. For LLM text, stream tokens via SSE — same total latency but feels significantly faster to the user. For image/video, show a progress bar that tracks average latency, then polls for completion. If generation crosses 15 seconds, offer email notification and let the user leave the page. Async generation should never block the funnel builder.

---

### Q5 · Three Challenges: Notebook → Production

**1. Data distribution drift** — Model accuracy degrades silently as real-world data shifts away from training data. Fix: log features per prediction, run weekly drift detection (Evidently AI), trigger retraining when drift exceeds threshold.

**2. Environment reproducibility** — Works locally, breaks in prod due to library version differences. Fix: Docker with pinned `requirements.txt`, MLflow/DVC to track the exact environment alongside the model artifact.

**3. Batch vs real-time latency** — Notebooks process one row at a time with no latency constraints. Production needs sub-200ms at 100+ concurrent requests. Fix: export to ONNX (5–10x speedup), add request batching in FastAPI, cache predictions for identical inputs in Redis.

---

### Q6 · LoRA for Face Consistency

LoRA fine-tunes a small adapter on ~20 reference photos using DreamBooth, layered on a frozen SDXL base. Training: ~15 min, ~$0.50–$0.80 on Replicate or Modal.com. Output: `.safetensors` file stored on S3, linked to user account.

At inference: pass `lora_weights` URL + `lora_scale` (default 0.82) + trigger word in prompt. Cache LoRA file per user (24h TTL). Key params: `lora_scale` 0.7–0.9, `num_inference_steps` 30–40, `guidance_scale` 7–8.

---

### Q7 · Tools & Frameworks

- **ML:** LightGBM + SHAP (churn prediction), scikit-learn, Prophet + XGBoost + SARIMA + LSTM (time-series forecasting)
- **MLOps:** FastAPI model serving, Docker, MLflow concepts
- **NLP:** sentence-transformers, TF-IDF from scratch (this assessment), BERT classifier
- **CV:** OpenCV, Grad-CAM for defect detection
- **Data engineering:** PySpark, Hive, Kafka, dbt with Medallion architecture
- **LLM APIs:** Gemini 1.5 Flash — automated YouTube channel pipeline
- **Python:** pandas, numpy, FastAPI, MoviePy, openpyxl, Playwright

Full portfolio: **[github.com/karans22](https://github.com/karans22)** — 14 repositories across data analyst, data engineer, ML engineer, and data scientist tracks.
