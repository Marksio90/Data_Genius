-- ==========================================================
-- DataGenius PRO - Initial Schema
-- Portable SQL for SQLite / PostgreSQL
-- ==========================================================

-- SQLite only (ignores on Postgres)
PRAGMA foreign_keys = ON;

BEGIN;

-- =========================
-- CORE
-- =========================

CREATE TABLE IF NOT EXISTS users (
    id               TEXT PRIMARY KEY,                   -- UUID string
    email            TEXT UNIQUE,
    name             TEXT,
    settings_json    TEXT,                               -- user-level prefs
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    id               TEXT PRIMARY KEY,                   -- session UUID
    user_id          TEXT REFERENCES users(id) ON DELETE SET NULL,
    status           TEXT NOT NULL DEFAULT 'initialized',-- initialized|data_loaded|eda_complete|training|completed|failed
    meta_json        TEXT,                               -- misc info, environment, app version
    started_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at         TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

-- =========================
-- DATASETS & EDA
-- =========================

CREATE TABLE IF NOT EXISTS datasets (
    id               TEXT PRIMARY KEY,                   -- dataset UUID
    session_id       TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    name             TEXT NOT NULL,
    file_path        TEXT NOT NULL,
    file_type        TEXT,
    n_rows           INTEGER,
    n_columns        INTEGER,
    size_bytes       INTEGER,
    hash             TEXT,                               -- df hash
    preview_path     TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, name)
);

CREATE INDEX IF NOT EXISTS idx_datasets_session ON datasets(session_id);

CREATE TABLE IF NOT EXISTS eda_results (
    id               TEXT PRIMARY KEY,
    dataset_id       TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    summary_json     TEXT,                               -- executive summary, key findings
    statistics_json  TEXT,                               -- overall + per-feature num/cat/distributions
    outliers_json    TEXT,
    correlations_json TEXT,
    report_path      TEXT,                               -- generated report (html/pdf/md)
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_eda_dataset ON eda_results(dataset_id);

-- =========================
-- PIPELINES & ARTIFACTS
-- =========================

CREATE TABLE IF NOT EXISTS pipelines (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    name             TEXT NOT NULL,
    description      TEXT,
    config_json      TEXT,                               -- pipeline config (preprocessing/fe/encoders/scalers)
    status           TEXT NOT NULL DEFAULT 'created',    -- created|running|completed|failed
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, name)
);

CREATE INDEX IF NOT EXISTS idx_pipelines_session ON pipelines(session_id);

CREATE TABLE IF NOT EXISTS artifacts (
    id               TEXT PRIMARY KEY,
    session_id       TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    pipeline_id      TEXT REFERENCES pipelines(id) ON DELETE CASCADE,
    model_id         TEXT,
    kind             TEXT NOT NULL,                      -- 'model'|'transformer'|'report'|'plot'|'cache'
    file_path        TEXT NOT NULL,
    size_bytes       INTEGER,
    metadata_json    TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_artifacts_pipeline ON artifacts(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);

-- =========================
-- MODELS & TRAINING
-- =========================

CREATE TABLE IF NOT EXISTS models (
    id               TEXT PRIMARY KEY,                   -- model UUID
    session_id       TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    pipeline_id      TEXT REFERENCES pipelines(id) ON DELETE SET NULL,
    name             TEXT NOT NULL,                      -- human-friendly name
    problem_type     TEXT NOT NULL,                      -- classification|regression
    framework        TEXT DEFAULT 'pycaret',
    algorithm        TEXT,                               -- e.g. 'lightgbm', 'xgboost'
    params_json      TEXT,                               -- hyperparams
    model_path       TEXT,                               -- persisted model
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_models_session ON models(session_id);
CREATE INDEX IF NOT EXISTS idx_models_pipeline ON models(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_models_problem ON models(problem_type);

CREATE TABLE IF NOT EXISTS training_runs (
    id               TEXT PRIMARY KEY,
    model_id         TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    dataset_id       TEXT REFERENCES datasets(id) ON DELETE SET NULL,
    status           TEXT NOT NULL DEFAULT 'finished',   -- finished|failed|running
    metrics_json     TEXT,                               -- dict of metrics
    best_score       REAL,
    cv_folds         INTEGER,
    duration_seconds REAL,
    notes            TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runs_model ON training_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON training_runs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON training_runs(status);

CREATE TABLE IF NOT EXISTS training_metrics (
    id               TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    metric_name      TEXT NOT NULL,
    metric_value     REAL NOT NULL,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_tmetrics_run ON training_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_tmetrics_name ON training_metrics(metric_name);

CREATE TABLE IF NOT EXISTS feature_importances (
    id               TEXT PRIMARY KEY,
    model_id         TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    feature_name     TEXT NOT NULL,
    importance       REAL NOT NULL,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fi_model ON feature_importances(model_id);
CREATE INDEX IF NOT EXISTS idx_fi_model_importance ON feature_importances(model_id, importance);

-- =========================
-- INFERENCE / PREDICTIONS
-- =========================

CREATE TABLE IF NOT EXISTS predictions (
    id               TEXT PRIMARY KEY,
    model_id         TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    input_path       TEXT,
    output_path      TEXT,
    n_rows           INTEGER,
    metrics_json     TEXT,                               -- post-inference metrics if ground truth available
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_id);

-- =========================
-- MONITORING (DRIFT, PERFORMANCE, ALERTS)
-- =========================

CREATE TABLE IF NOT EXISTS monitoring_snapshots (
    id               TEXT PRIMARY KEY,
    model_id         TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    window_start     TIMESTAMP,
    window_end       TIMESTAMP,
    data_stats_json  TEXT,                               -- stats for window: counts, missing, etc.
    perf_metrics_json TEXT,                              -- performance over window
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ms_model ON monitoring_snapshots(model_id);
CREATE INDEX IF NOT EXISTS idx_ms_window ON monitoring_snapshots(window_start, window_end);

CREATE TABLE IF NOT EXISTS drift_events (
    id               TEXT PRIMARY KEY,
    model_id         TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    dataset_id       TEXT REFERENCES datasets(id) ON DELETE SET NULL,
    drift_type       TEXT NOT NULL,                      -- 'data'|'concept'
    detector         TEXT,                               -- psi|ks|js|custom
    metric_value     REAL,
    threshold        REAL,
    severity         TEXT,                               -- info|warning|critical
    details_json     TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_drift_model ON drift_events(model_id);
CREATE INDEX IF NOT EXISTS idx_drift_created ON drift_events(created_at);

CREATE TABLE IF NOT EXISTS performance_snapshots (
    id               TEXT PRIMARY KEY,
    model_id         TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    metric_name      TEXT NOT NULL,                      -- accuracy|f1|r2|rmse etc.
    value            REAL NOT NULL,
    baseline_value   REAL,
    delta            REAL,                               -- value - baseline
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_perf_model_metric ON performance_snapshots(model_id, metric_name);

CREATE TABLE IF NOT EXISTS alerts (
    id               TEXT PRIMARY KEY,
    related_model_id TEXT REFERENCES models(id) ON DELETE SET NULL,
    related_event_id TEXT,                               -- drift/perf event id (free text/foreign key optional)
    event_type       TEXT NOT NULL,                      -- drift|performance|system
    severity         TEXT NOT NULL DEFAULT 'info',       -- info|warning|critical
    message          TEXT NOT NULL,
    channel          TEXT,                               -- email|slack|log
    status           TEXT NOT NULL DEFAULT 'new',        -- new|sent|ack|closed
    payload_json     TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_model ON alerts(related_model_id);

-- =========================
-- AI MENTOR CHAT
-- =========================

CREATE TABLE IF NOT EXISTS chat_messages (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role             TEXT NOT NULL,                      -- user|assistant|system
    content          TEXT NOT NULL,
    tokens_used      INTEGER,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_created ON chat_messages(created_at);

-- =========================
-- REPORTS
-- =========================

CREATE TABLE IF NOT EXISTS reports (
    id               TEXT PRIMARY KEY,
    session_id       TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    model_id         TEXT REFERENCES models(id) ON DELETE SET NULL,
    dataset_id       TEXT REFERENCES datasets(id) ON DELETE SET NULL,
    report_type      TEXT NOT NULL,                      -- eda|training|monitoring|executive
    format           TEXT NOT NULL,                      -- html|pdf|markdown|docx
    file_path        TEXT NOT NULL,
    metadata_json    TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reports_model ON reports(model_id);
CREATE INDEX IF NOT EXISTS idx_reports_session ON reports(session_id);

-- =========================
-- VIEWS (helpers)
-- =========================

-- Ostatni znany wynik metryki per model
CREATE VIEW IF NOT EXISTS vw_model_latest_score AS
SELECT
    m.id            AS model_id,
    m.name          AS model_name,
    tr.best_score   AS best_score,
    tr.created_at   AS evaluated_at
FROM models m
LEFT JOIN (
    SELECT model_id, best_score, created_at
    FROM training_runs
    WHERE created_at = (
        SELECT MAX(created_at) FROM training_runs tr2 WHERE tr2.model_id = training_runs.model_id
    )
) tr ON tr.model_id = m.id;

-- Najważniejsze cechy (top N per model można dociąć zapytaniem)
CREATE VIEW IF NOT EXISTS vw_feature_importances AS
SELECT
    model_id,
    feature_name,
    importance
FROM feature_importances
ORDER BY model_id, importance DESC;

COMMIT;

-- ==========================================================
-- Notes:
-- * ID są typu TEXT/UUID – generuj po stronie aplikacji (np. uuid4()).
-- * JSON przechowujemy jako TEXT (Postgres potraktuje to jako TEXT/JSONB po migracjach).
-- * PRAGMA foreign_keys włącza FK w SQLite.
-- ==========================================================
