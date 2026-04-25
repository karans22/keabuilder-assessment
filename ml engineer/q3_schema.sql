-- q3_schema.sql
--
-- KeaBuilder ML Engineer Assessment -- Q3
-- Schema for storing user inputs and ML predictions.
--
-- Assumes PostgreSQL with the pgvector extension for embedding storage.
-- Run: CREATE EXTENSION IF NOT EXISTS vector;  before applying this.
--
-- The two tables are intentionally separate: user_inputs captures what came
-- in, ml_predictions captures what the model decided. This way we can re-run
-- predictions on old inputs when we ship a new model version, and compare
-- results without touching the original data.


-- ---------------------------------------------------------------------------
-- user_inputs
-- Everything that arrives from a user: form submissions, chatbot messages,
-- prompt strings, etc.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS user_inputs (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID            NOT NULL,
    funnel_id       UUID,                           -- nullable for chatbot/prompt inputs
    input_text      TEXT            NOT NULL,        -- raw text as the user typed it
    input_vector    VECTOR(384),                    -- MiniLM-L6 embedding, NULL until bg job runs
    input_type      VARCHAR(50)     NOT NULL
                    CHECK (input_type IN ('lead_form', 'chatbot_message', 'prompt')),
    metadata        JSONB           DEFAULT '{}',   -- UTM, browser, device, anything extra
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- fast lookup by user and funnel
CREATE INDEX IF NOT EXISTS idx_user_inputs_user_id
    ON user_inputs (user_id);

CREATE INDEX IF NOT EXISTS idx_user_inputs_funnel_id
    ON user_inputs (funnel_id)
    WHERE funnel_id IS NOT NULL;

-- HNSW index for approximate nearest-neighbor search
-- ~1% recall tradeoff but query time drops from seconds to milliseconds at scale
CREATE INDEX IF NOT EXISTS idx_user_inputs_vector
    ON user_inputs USING hnsw (input_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);


-- ---------------------------------------------------------------------------
-- ml_predictions
-- One row per model inference. Stores a feature snapshot so we can audit
-- and debug model behavior later, especially after retraining.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ml_predictions (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    input_id            UUID        NOT NULL REFERENCES user_inputs (id) ON DELETE CASCADE,
    model_name          VARCHAR(100) NOT NULL,   -- lead_classifier | churn_model | etc.
    model_version       VARCHAR(20)  NOT NULL,   -- semver e.g. 1.2.0
    prediction_label    VARCHAR(50)  NOT NULL,   -- HOT | WARM | COLD | churn | no_churn
    confidence          FLOAT        NOT NULL
                        CHECK (confidence BETWEEN 0 AND 1),
    all_probabilities   JSONB,                   -- {"HOT": 0.82, "WARM": 0.14, "COLD": 0.04}
    feature_snapshot    JSONB        NOT NULL,   -- exact features used at inference time
    latency_ms          INTEGER,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_input_id
    ON ml_predictions (input_id);

CREATE INDEX IF NOT EXISTS idx_predictions_model
    ON ml_predictions (model_name, model_version, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_label
    ON ml_predictions (prediction_label, created_at DESC);


-- ---------------------------------------------------------------------------
-- useful queries
-- ---------------------------------------------------------------------------

-- 1. find the 10 most similar leads to a new input vector
--    (pass the query embedding as $1 from your application)
--
-- SELECT id, input_text,
--        1 - (input_vector <=> $1::vector) AS similarity
-- FROM   user_inputs
-- WHERE  user_id    = $2
--   AND  input_type = 'lead_form'
-- ORDER  BY input_vector <=> $1::vector
-- LIMIT  10;


-- 2. latest prediction per lead, last 7 days
--
-- SELECT DISTINCT ON (ui.id)
--        ui.id, ui.input_text,
--        p.prediction_label, p.confidence, p.created_at
-- FROM   user_inputs   ui
-- JOIN   ml_predictions p ON p.input_id = ui.id
-- WHERE  ui.user_id    = $1
--   AND  ui.input_type = 'lead_form'
--   AND  ui.created_at >= NOW() - INTERVAL '7 days'
-- ORDER  BY ui.id, p.created_at DESC;


-- 3. weekly HOT/WARM/COLD breakdown for a given model
--
-- SELECT DATE_TRUNC('week', created_at)  AS week,
--        prediction_label,
--        COUNT(*)                         AS count,
--        ROUND(COUNT(*) * 100.0 /
--              SUM(COUNT(*)) OVER (PARTITION BY DATE_TRUNC('week', created_at)), 1) AS pct
-- FROM   ml_predictions
-- WHERE  model_name = 'lead_classifier'
-- GROUP  BY 1, 2
-- ORDER  BY 1, 2;
