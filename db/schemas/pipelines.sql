-- ==========================================================
-- DataGenius PRO - Pipelines Schema
-- Definicje, kroki, uruchomienia, artefakty, harmonogramy
-- ==========================================================

PRAGMA foreign_keys = ON; -- ignorowane przez Postgres
BEGIN;

-- ----------------------------------------------------------
-- 1) Definicje pipeline'ów (szablony/konfiguracje)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_definitions (
    id                 TEXT PRIMARY KEY,                     -- uuid
    name               TEXT NOT NULL,
    version            TEXT NOT NULL DEFAULT '1.0.0',
    description        TEXT,
    tags               TEXT,                                 -- CSV lub JSON (TEXT)
    default_params_json TEXT,                                -- parametry domyślne
    owner              TEXT,                                 -- np. użytkownik/serwis
    is_active          INTEGER NOT NULL DEFAULT 1,           -- bool
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version)
);

CREATE INDEX IF NOT EXISTS idx_pdef_active ON pipeline_definitions(is_active);
CREATE INDEX IF NOT EXISTS idx_pdef_name ON pipeline_definitions(name);

-- ----------------------------------------------------------
-- 2) Kroki pipeline'u (kolejność + konfiguracja agenta)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_steps (
    id                 TEXT PRIMARY KEY,
    pipeline_id        TEXT NOT NULL REFERENCES pipeline_definitions(id) ON DELETE CASCADE,
    name               TEXT NOT NULL,                        -- np. 'EDA', 'Training'
    agent_name         TEXT NOT NULL,                        -- np. 'ModelTrainer'
    step_order         INTEGER NOT NULL,                     -- kolejnosc
    config_json        TEXT,                                 -- parametry kroku/agentów
    retry_policy_json  TEXT,                                 -- max_retries, backoff itp.
    enabled            INTEGER NOT NULL DEFAULT 1,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pipeline_id, name),
    UNIQUE(pipeline_id, step_order)
);

CREATE INDEX IF NOT EXISTS idx_psteps_pipeline ON pipeline_steps(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_psteps_enabled ON pipeline_steps(enabled);

-- (opcjonalnie) zależności między krokami (DAG)
CREATE TABLE IF NOT EXISTS step_dependencies (
    id                 TEXT PRIMARY KEY,
    pipeline_id        TEXT NOT NULL REFERENCES pipeline_definitions(id) ON DELETE CASCADE,
    step_id            TEXT NOT NULL REFERENCES pipeline_steps(id) ON DELETE CASCADE,
    depends_on_step_id TEXT NOT NULL REFERENCES pipeline_steps(id) ON DELETE CASCADE,
    UNIQUE(step_id, depends_on_step_id)
);

CREATE INDEX IF NOT EXISTS idx_sdeps_pipeline ON step_dependencies(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_sdeps_step ON step_dependencies(step_id);

-- ----------------------------------------------------------
-- 3) Uruchomienia pipeline'u (run)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id                 TEXT PRIMARY KEY,
    pipeline_id        TEXT NOT NULL REFERENCES pipeline_definitions(id) ON DELETE CASCADE,
    session_id         TEXT,                                 -- z StateManager
    trigger_type       TEXT NOT NULL DEFAULT 'manual',       -- manual|schedule|api|retrain
    status             TEXT NOT NULL DEFAULT 'started',      -- started|running|success|failed|cancelled|partial
    started_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at        TIMESTAMP,
    params_json        TEXT,                                 -- parametry dla biegu
    context_json       TEXT,                                 -- np. ścieżki danych, meta
    metrics_json       TEXT,                                 -- zbiorcze metryki
    error_message      TEXT,
    data_hash          TEXT                                  -- hash wejściowego datasetu
);

CREATE INDEX IF NOT EXISTS idx_pruns_pipeline ON pipeline_runs(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_pruns_status ON pipeline_runs(status);
CREATE INDEX IF NOT EXISTS idx_pruns_started ON pipeline_runs(started_at);

-- ----------------------------------------------------------
-- 4) Uruchomienia kroków (step runs)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS step_runs (
    id                   TEXT PRIMARY KEY,
    pipeline_run_id      TEXT NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    step_id              TEXT NOT NULL REFERENCES pipeline_steps(id) ON DELETE SET NULL,
    agent_name           TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'started',    -- started|running|success|failed|skipped|retrying
    attempt              INTEGER NOT NULL DEFAULT 1,
    started_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at          TIMESTAMP,
    input_artifacts_json TEXT,
    output_artifacts_json TEXT,
    metrics_json         TEXT,
    warnings_json        TEXT,
    error_message        TEXT,
    log_path             TEXT                                -- ścieżka do logów/artefaktów
);

CREATE INDEX IF NOT EXISTS idx_srun_prun ON step_runs(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_srun_status ON step_runs(status);
CREATE INDEX IF NOT EXISTS idx_srun_step ON step_runs(step_id);

-- ----------------------------------------------------------
-- 5) Artefakty (modele, raporty, dane pośrednie)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS artifacts (
    id                 TEXT PRIMARY KEY,
    pipeline_run_id    TEXT REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    step_run_id        TEXT REFERENCES step_runs(id) ON DELETE SET NULL,
    type               TEXT NOT NULL,            -- dataset|model|report|figure|metrics|cache|other
    name               TEXT NOT NULL,
    uri                TEXT,                     -- ścieżka pliku/URL
    format             TEXT,                     -- csv|parquet|pkl|html|pdf|json|png...
    size_bytes         INTEGER,
    hash               TEXT,
    metadata_json      TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_art_prun ON artifacts(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_art_step ON artifacts(step_run_id);
CREATE INDEX IF NOT EXISTS idx_art_type ON artifacts(type);

-- ----------------------------------------------------------
-- 6) Harmonogramy pipeline'ów
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_schedules (
    id                 TEXT PRIMARY KEY,
    pipeline_id        TEXT NOT NULL REFERENCES pipeline_definitions(id) ON DELETE CASCADE,
    schedule           TEXT NOT NULL,            -- daily|weekly|cron:* * * * *|@every 1h
    timezone           TEXT DEFAULT 'Europe/Warsaw',
    params_json        TEXT,
    enabled            INTEGER NOT NULL DEFAULT 1,
    last_run_at        TIMESTAMP,
    next_run_at        TIMESTAMP,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_psched_pipeline ON pipeline_schedules(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_psched_enabled ON pipeline_schedules(enabled);
CREATE INDEX IF NOT EXISTS idx_psched_next ON pipeline_schedules(next_run_at);

-- ----------------------------------------------------------
-- 7) Kolejka uruchomień (dla scheduler/worker)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_queue (
    id                 TEXT PRIMARY KEY,
    pipeline_id        TEXT NOT NULL REFERENCES pipeline_definitions(id) ON DELETE CASCADE,
    pipeline_run_id    TEXT REFERENCES pipeline_runs(id) ON DELETE SET NULL,
    priority           INTEGER NOT NULL DEFAULT 5, -- 1 (wysoki) - 9 (niski)
    status             TEXT NOT NULL DEFAULT 'queued', -- queued|running|succeeded|failed|cancelled
    scheduled_for      TIMESTAMP,
    started_at         TIMESTAMP,
    finished_at        TIMESTAMP,
    retries            INTEGER NOT NULL DEFAULT 0,
    max_retries        INTEGER NOT NULL DEFAULT 3,
    params_json        TEXT,
    error_message      TEXT
);

CREATE INDEX IF NOT EXISTS idx_pq_status ON pipeline_queue(status);
CREATE INDEX IF NOT EXISTS idx_pq_sched ON pipeline_queue(scheduled_for);
CREATE INDEX IF NOT EXISTS idx_pq_pipeline ON pipeline_queue(pipeline_id);

-- ----------------------------------------------------------
-- 8) Zdarzenia/logi pipeline (lekki event store)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_events (
    id                 TEXT PRIMARY KEY,
    pipeline_run_id    TEXT REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    step_run_id        TEXT REFERENCES step_runs(id) ON DELETE SET NULL,
    level              TEXT NOT NULL DEFAULT 'INFO', -- DEBUG|INFO|WARN|ERROR
    message            TEXT NOT NULL,
    context_json       TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pev_prun ON pipeline_events(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_pev_step ON pipeline_events(step_run_id);
CREATE INDEX IF NOT EXISTS idx_pev_level ON pipeline_events(level);

-- ----------------------------------------------------------
-- 9) Widoki pomocnicze
-- ----------------------------------------------------------
-- Ostatni run per pipeline
CREATE VIEW IF NOT EXISTS vw_latest_pipeline_runs AS
SELECT pr.*
FROM pipeline_runs pr
JOIN (
    SELECT pipeline_id, MAX(started_at) AS max_started
    FROM pipeline_runs
    GROUP BY pipeline_id
) last ON last.pipeline_id = pr.pipeline_id AND last.max_started = pr.started_at;

-- Skrót statusów kroków per run
CREATE VIEW IF NOT EXISTS vw_step_run_summary AS
SELECT
    pr.id AS pipeline_run_id,
    COUNT(sr.id) AS steps_total,
    SUM(CASE WHEN sr.status='success' THEN 1 ELSE 0 END) AS steps_success,
    SUM(CASE WHEN sr.status='failed'  THEN 1 ELSE 0 END) AS steps_failed,
    SUM(CASE WHEN sr.status='skipped' THEN 1 ELSE 0 END) AS steps_skipped
FROM pipeline_runs pr
LEFT JOIN step_runs sr ON sr.pipeline_run_id = pr.id
GROUP BY pr.id;

-- ----------------------------------------------------------
-- 10) SEED: przykładowy pipeline "tabular_default"
-- ----------------------------------------------------------
INSERT OR IGNORE INTO pipeline_definitions
(id, name, version, description, tags, default_params_json, owner, is_active)
VALUES
('tabular_default_v1',
 'Tabular Default',
 '1.0.0',
 'Domyślny pipeline: upload -> EDA -> preprocessing -> training -> evaluation -> report',
 '["tabular","automl","eda"]',
 '{"test_size":0.2,"cv_folds":5,"enable_tuning":true}',
 'system',
 1
);

-- Kroki (kolejność)
INSERT OR IGNORE INTO pipeline_steps
(id, pipeline_id, name, agent_name, step_order, config_json, enabled)
VALUES
('step_upload',  'tabular_default_v1', 'Data Upload',    'FileHandler',       1, '{}', 1),
('step_eda',     'tabular_default_v1', 'EDA',            'EDAOrchestrator',   2, '{}', 1),
('step_prep',    'tabular_default_v1', 'Preprocessing',  'PipelineBuilder',   3, '{}', 1),
('step_train',   'tabular_default_v1', 'Training',       'MLOrchestrator',    4, '{}', 1),
('step_eval',    'tabular_default_v1', 'Evaluation',     'ModelEvaluator',    5, '{}', 1),
('step_report',  'tabular_default_v1', 'Report',         'ReportGenerator',   6, '{}', 1);

-- Zależności DAG (prosta ścieżka liniowa)
INSERT OR IGNORE INTO step_dependencies (id, pipeline_id, step_id, depends_on_step_id) VALUES
('dep_eda',    'tabular_default_v1', 'step_eda',   'step_upload'),
('dep_prep',   'tabular_default_v1', 'step_prep',  'step_eda'),
('dep_train',  'tabular_default_v1', 'step_train', 'step_prep'),
('dep_eval',   'tabular_default_v1', 'step_eval',  'step_train'),
('dep_report', 'tabular_default_v1', 'step_report','step_eval');

COMMIT;

-- ==========================================================
-- Notatki:
-- * BOOLEAN reprezentowany jako INTEGER 1/0 (SQLite). W Postgres
--   można zcastować do BOOLEAN w późniejszych migracjach.
-- * Pola *_json pozostają TEXT (w Postgres można zmienić na JSONB).
-- * Widoki ułatwiają dashboard i szybkie podsumowania.
-- * Kolejka + harmonogram = wsparcie dla workerów/schedulera.
-- ==========================================================
