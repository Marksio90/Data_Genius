-- ==========================================================
-- DataGenius PRO - Monitoring Schema (drift + performance)
-- Portable SQL for SQLite / PostgreSQL
-- ==========================================================

PRAGMA foreign_keys = ON; -- ignored by Postgres
BEGIN;

-- ----------------------------------------------------------
-- 0) Helpers: updated_at triggers (SQLite-friendly)
-- ----------------------------------------------------------
-- Uwaga: w Postgres można dodać osobne trigger funkcje; tutaj
-- pozostawiamy proste aktualizacje w aplikacji lub dopisujemy
-- triggery per tabela (poniżej gdzie sensowne).

-- ----------------------------------------------------------
-- 1) Deployments: instancje modeli pod monitorowanie
--    (można spiąć z model_registry.id)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_deployments (
    id               TEXT PRIMARY KEY,                 -- np. uuid
    registry_id      TEXT REFERENCES model_registry(id) ON DELETE SET NULL,
    model_version    TEXT NOT NULL,                    -- np. 'v1.3.2'
    problem_type     TEXT NOT NULL,                    -- classification|regression
    target_column    TEXT,
    environment      TEXT DEFAULT 'production',        -- dev|staging|production
    serving_uri      TEXT,
    metadata_json    TEXT,                             -- dowolne meta
    active           INTEGER NOT NULL DEFAULT 1,       -- bool 1/0
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(registry_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_deploy_active ON model_deployments(active);
CREATE INDEX IF NOT EXISTS idx_deploy_problem ON model_deployments(problem_type);

-- ----------------------------------------------------------
-- 2) Konfiguracja progów monitoringu (globalna lub per deployment)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS monitoring_config (
    id                   TEXT PRIMARY KEY,                  -- 'default' lub uuid
    deployment_id        TEXT REFERENCES model_deployments(id) ON DELETE CASCADE,
    psi_threshold        REAL  NOT NULL DEFAULT 0.10,       -- Population Stability Index
    ks_threshold         REAL  NOT NULL DEFAULT 0.05,       -- Kolmogorov–Smirnov p-value*
    js_threshold         REAL  NOT NULL DEFAULT 0.10,       -- Jensen–Shannon
    perf_drop_threshold  REAL  NOT NULL DEFAULT 0.05,       -- dopuszczalny spadek metryki (5%)
    window_days          INTEGER NOT NULL DEFAULT 7,        -- rozmiar okna danych produkcyjnych
    schedule             TEXT NOT NULL DEFAULT 'weekly',    -- daily|weekly|monthly|cron:...
    enabled              INTEGER NOT NULL DEFAULT 1,        -- bool
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_moncfg_deploy ON monitoring_config(deployment_id);
CREATE INDEX IF NOT EXISTS idx_moncfg_enabled ON monitoring_config(enabled);

-- ----------------------------------------------------------
-- 3) Definicje przekrojów/slice do monitorowania (np. region='EU')
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS data_slices (
    id               TEXT PRIMARY KEY,
    deployment_id    TEXT NOT NULL REFERENCES model_deployments(id) ON DELETE CASCADE,
    name             TEXT NOT NULL,                        -- np. 'EU', 'new_users'
    definition_json  TEXT NOT NULL,                        -- filtr w JSON (expr/DSL)
    enabled          INTEGER NOT NULL DEFAULT 1,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(deployment_id, name)
);

CREATE INDEX IF NOT EXISTS idx_slices_deploy ON data_slices(deployment_id);
CREATE INDEX IF NOT EXISTS idx_slices_enabled ON data_slices(enabled);

-- ----------------------------------------------------------
-- 4) Monitoring runs (snapshot z okna danych i referencji)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS monitoring_runs (
    id                   TEXT PRIMARY KEY,
    deployment_id        TEXT NOT NULL REFERENCES model_deployments(id) ON DELETE CASCADE,
    status               TEXT NOT NULL DEFAULT 'started',   -- started|success|failed
    started_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at          TIMESTAMP,
    window_start         TIMESTAMP,                         -- dane produkcyjne (okno)
    window_end           TIMESTAMP,
    ref_start            TIMESTAMP,                         -- okno referencyjne (trening / baseline)
    ref_end              TIMESTAMP,
    row_count            INTEGER DEFAULT 0,
    data_hash            TEXT,                              -- hash batcha
    dataset_stats_json   TEXT,                              -- ogólny profil danych
    error_message        TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_deploy ON monitoring_runs(deployment_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON monitoring_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_window ON monitoring_runs(window_end);

-- ----------------------------------------------------------
-- 5) Statystyki cech i metryki driftu per run + cecha
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS feature_drift (
    id                 TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES monitoring_runs(id) ON DELETE CASCADE,
    feature_name       TEXT NOT NULL,
    feature_type       TEXT,                    -- numeric|categorical|datetime|text
    psi                REAL,                    -- Population Stability Index
    ks_stat            REAL,                    -- KS statistic
    ks_pvalue          REAL,
    js_divergence      REAL,
    wasserstein        REAL,
    chi2_pvalue        REAL,                    -- dla kategorii
    missing_pct        REAL,
    unique_count       INTEGER,
    ref_stats_json     TEXT,                    -- mean/std/topk...
    cur_stats_json     TEXT,
    thresholds_json    TEXT,                    -- zapis progów użytych w tym biegu
    is_drift           INTEGER NOT NULL DEFAULT 0,  -- bool (wg progów)
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, feature_name)
);

CREATE INDEX IF NOT EXISTS idx_fd_run ON feature_drift(run_id);
CREATE INDEX IF NOT EXISTS idx_fd_drift ON feature_drift(is_drift);

-- ----------------------------------------------------------
-- 6) Metryki jakości/predykcji (globalnie i dla slice)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS performance_metrics (
    id                 TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES monitoring_runs(id) ON DELETE CASCADE,
    slice_id           TEXT REFERENCES data_slices(id) ON DELETE SET NULL,
    metric_name        TEXT NOT NULL,               -- accuracy|f1|auc|r2|mae|...
    value              REAL NOT NULL,
    baseline_value     REAL,                        -- z treningu lub referencji
    delta              REAL,                        -- value - baseline (lub -+)
    higher_is_better   INTEGER NOT NULL DEFAULT 1,  -- bool
    is_degradation     INTEGER NOT NULL DEFAULT 0,  -- przekroczony próg spadku
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, COALESCE(slice_id, 'GLOBAL'), metric_name)
);

CREATE INDEX IF NOT EXISTS idx_perf_run ON performance_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_perf_slice ON performance_metrics(slice_id);
CREATE INDEX IF NOT EXISTS idx_perf_deg ON performance_metrics(is_degradation);

-- ----------------------------------------------------------
-- 7) Alerty
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    id                 TEXT PRIMARY KEY,
    deployment_id      TEXT NOT NULL REFERENCES model_deployments(id) ON DELETE CASCADE,
    run_id             TEXT REFERENCES monitoring_runs(id) ON DELETE SET NULL,
    type               TEXT NOT NULL,                -- drift|performance|data_quality|system
    severity           TEXT NOT NULL DEFAULT 'medium', -- low|medium|high|critical
    title              TEXT NOT NULL,
    message            TEXT,
    context_json       TEXT,
    status             TEXT NOT NULL DEFAULT 'open',   -- open|ack|closed
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at          TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_deploy ON alerts(deployment_id);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(type);

-- ----------------------------------------------------------
-- 8) Notyfikacje (kanały wyjściowe)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS notifications (
    id                 TEXT PRIMARY KEY,
    alert_id           TEXT NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    channel            TEXT NOT NULL,            -- email|slack|webhook
    target             TEXT,                     -- adres email / url webhooka
    payload_json       TEXT,
    status             TEXT NOT NULL DEFAULT 'sent', -- sent|failed
    error_message      TEXT,
    sent_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_notif_alert ON notifications(alert_id);

-- ----------------------------------------------------------
-- 9) Kolejka retrainingu
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS retraining_queue (
    id                 TEXT PRIMARY KEY,
    deployment_id      TEXT NOT NULL REFERENCES model_deployments(id) ON DELETE CASCADE,
    trigger_type       TEXT NOT NULL,              -- drift|performance|manual|schedule
    run_id             TEXT REFERENCES monitoring_runs(id) ON DELETE SET NULL,
    priority           INTEGER NOT NULL DEFAULT 5, -- 1 (wysoki) - 9 (niski)
    status             TEXT NOT NULL DEFAULT 'queued', -- queued|running|succeeded|failed|cancelled
    scheduled_for      TIMESTAMP,
    started_at         TIMESTAMP,
    finished_at        TIMESTAMP,
    params_json        TEXT,                       -- np. zakres danych, feature flags
    notes              TEXT
);

CREATE INDEX IF NOT EXISTS idx_retrain_deploy ON retraining_queue(deployment_id);
CREATE INDEX IF NOT EXISTS idx_retrain_status ON retraining_queue(status);
CREATE INDEX IF NOT EXISTS idx_retrain_sched ON retraining_queue(scheduled_for);

-- ----------------------------------------------------------
-- 10) Widoki pomocnicze
-- ----------------------------------------------------------

-- Ostatni run per deployment
CREATE VIEW IF NOT EXISTS vw_latest_run_per_deployment AS
SELECT mr.*
FROM monitoring_runs mr
JOIN (
    SELECT deployment_id, MAX(started_at) AS max_start
    FROM monitoring_runs
    GROUP BY deployment_id
) latest
ON latest.deployment_id = mr.deployment_id
AND latest.max_start = mr.started_at;

-- Podsumowanie driftu per run
CREATE VIEW IF NOT EXISTS vw_drift_summary AS
SELECT
    fd.run_id,
    COUNT(*)                      AS features_checked,
    SUM(CASE WHEN fd.is_drift=1 THEN 1 ELSE 0 END) AS features_drifted,
    ROUND(100.0 * SUM(CASE WHEN fd.is_drift=1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) AS drift_pct
FROM feature_drift fd
GROUP BY fd.run_id;

-- Otwarte alerty z tytułem + środowisko
CREATE VIEW IF NOT EXISTS vw_open_alerts AS
SELECT
    a.id,
    a.type,
    a.severity,
    a.title,
    a.created_at,
    d.environment,
    d.id AS deployment_id
FROM alerts a
JOIN model_deployments d ON d.id = a.deployment_id
WHERE a.status = 'open'
ORDER BY a.severity DESC, a.created_at DESC;

-- ==========================================================
-- SEED: domyślna konfiguracja (globalna)
-- ==========================================================
INSERT OR IGNORE INTO monitoring_config
(id, deployment_id, psi_threshold, ks_threshold, js_threshold, perf_drop_threshold, window_days, schedule, enabled)
VALUES
('default', NULL, 0.10, 0.05, 0.10, 0.05, 7, 'weekly', 1);

COMMIT;

-- ==========================================================
-- Notatki:
-- * BOOLEAN -> używamy INTEGER 1/0 (SQLite). W Postgres można
--   zrobić ALTER COLUMN na BOOLEAN.
-- * ks_threshold traktujemy jako próg p-value (niska wartość = drift).
-- * Pola *_json są TEXT; w Postgres można zmigrować do JSONB.
-- * Widoki ułatwiają dashboard/raporty: ostatnie runy, drift summary,
--   otwarte alerty. Dodatkowe indeksy można dobrać po profilu zapytań.
-- ==========================================================
