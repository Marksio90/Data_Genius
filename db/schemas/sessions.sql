-- ==========================================================
-- DataGenius PRO - Sessions Schema
-- Zarządzanie sesjami, danymi, czatem i stanem aplikacji
-- ==========================================================

PRAGMA foreign_keys = ON; -- ignorowane przez Postgres
BEGIN;

-- ----------------------------------------------------------
-- 1) Sesje aplikacyjne
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id               TEXT PRIMARY KEY,                 -- session_id (np. z StateManager)
    user_id          TEXT,                             -- opcjonalnie: ID użytkownika (gdy multi-user)
    status           TEXT NOT NULL DEFAULT 'initialized',  -- initialized|data_loaded|eda_complete|training|completed|failed
    pipeline_stage   TEXT DEFAULT 'initialized',       -- zgodne z PIPELINE_STAGES
    data_hash        TEXT,                             -- hash aktualnego datasetu
    target_column    TEXT,
    problem_type     TEXT,                             -- classification|regression|...
    app_version      TEXT,
    client_ip        TEXT,
    user_agent       TEXT,
    is_active        INTEGER NOT NULL DEFAULT 1,       -- bool 1/0
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at        TIMESTAMP,
    metadata_json    TEXT                              -- dowolne meta (TEXT/JSON)
);

CREATE INDEX IF NOT EXISTS idx_sessions_status       ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_active       ON sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_sessions_pipeline     ON sessions(pipeline_stage);
CREATE INDEX IF NOT EXISTS idx_sessions_created      ON sessions(created_at);

-- ----------------------------------------------------------
-- 2) Dane/Pliki w ramach sesji (uploady)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS session_datasets (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    name             TEXT,                             -- nazwa logiczna (np. pliku)
    path             TEXT,                             -- ścieżka do pliku (lokalna/S3/GCS)
    file_type        TEXT,                             -- csv|xlsx|json|parquet
    size_bytes       INTEGER,
    data_hash        TEXT,
    n_rows           INTEGER,
    n_columns        INTEGER,
    columns_json     TEXT,                             -- lista kolumn i dtypów
    preview_path     TEXT,                             -- zapis podglądu (opcjonalnie)
    is_current       INTEGER NOT NULL DEFAULT 1,       -- czy aktywny dataset sesji
    is_sample        INTEGER NOT NULL DEFAULT 0,       -- czy to dataset przykładowy
    uploaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json    TEXT
);

CREATE INDEX IF NOT EXISTS idx_sds_session     ON session_datasets(session_id);
CREATE INDEX IF NOT EXISTS idx_sds_current     ON session_datasets(is_current);
CREATE INDEX IF NOT EXISTS idx_sds_hash        ON session_datasets(data_hash);

-- ----------------------------------------------------------
-- 3) Snapshoty stanu sesji (dla debug/odtworzeń)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS session_state_snapshots (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    stage            TEXT NOT NULL,                    -- np. data_loaded/eda/preprocessing/training...
    state_json       TEXT NOT NULL,                    -- zserializowany stan (bez ciężkich binariów)
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sss_session     ON session_state_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_sss_stage       ON session_state_snapshots(stage);
CREATE INDEX IF NOT EXISTS idx_sss_created     ON session_state_snapshots(created_at);

-- ----------------------------------------------------------
-- 4) Historia czatu i LLM (AI Mentor)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS session_chats (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role             TEXT NOT NULL,                    -- user|assistant|system
    content          TEXT NOT NULL,                    -- treść wiadomości
    model            TEXT,                             -- użyty model (np. claude-sonnet-*, gpt-*)
    tokens_in        INTEGER,
    tokens_out       INTEGER,
    finish_reason    TEXT,
    metadata_json    TEXT,                             -- np. temperatury, prompty systemowe
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_schat_session   ON session_chats(session_id);
CREATE INDEX IF NOT EXISTS idx_schat_created   ON session_chats(created_at);
CREATE INDEX IF NOT EXISTS idx_schat_role      ON session_chats(role);

-- ----------------------------------------------------------
-- 5) Zdarzenia sesyjne (lekki event store)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS session_events (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    level            TEXT NOT NULL DEFAULT 'INFO',     -- DEBUG|INFO|WARN|ERROR
    source           TEXT,                             -- np. FileHandler, EDAOrchestrator
    message          TEXT NOT NULL,
    context_json     TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sev_session     ON session_events(session_id);
CREATE INDEX IF NOT EXISTS idx_sev_level       ON session_events(level);
CREATE INDEX IF NOT EXISTS idx_sev_created     ON session_events(created_at);

-- ----------------------------------------------------------
-- 6) Metryki i podsumowania sesji
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS session_metrics (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    data_quality_score   REAL,                         -- 0..100
    data_quality_json    TEXT,                         -- szczegóły (completeness/uniqueness/...)
    eda_summary_json     TEXT,                         -- skrót EDA (najważniejsze wnioski)
    model_summary_json   TEXT,                         -- skrót ML (best model, wyniki)
    recommendations_json TEXT,                         -- rekomendacje AI Mentora
    updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_smetrics_session ON session_metrics(session_id);

-- ----------------------------------------------------------
-- 7) Powiązania z artifactami i biegami pipeline (cross-module)
--    (artefakty są zdefiniowane w pipelines.sql)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS session_artifacts (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    artifact_id      TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    linked_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    note             TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_sart_session_artifact ON session_artifacts(session_id, artifact_id);

-- Pipeline runy mogą być już połączone z session_id w pipelines.sql (kolumna session_id w pipeline_runs).
-- Dla wygody tworzymy widok sesja -> ostatni bieg pipeline’u.

-- ----------------------------------------------------------
-- 8) Widoki pomocnicze
-- ----------------------------------------------------------
-- Ostatni snapshot stanu per sesja
CREATE VIEW IF NOT EXISTS vw_session_latest_state AS
SELECT s.id AS session_id,
       s.status,
       s.pipeline_stage,
       ss.stage AS snapshot_stage,
       ss.state_json,
       ss.created_at AS snapshot_created_at
FROM sessions s
LEFT JOIN (
    SELECT session_id,
           MAX(created_at) AS max_created
    FROM session_state_snapshots
    GROUP BY session_id
) last ON last.session_id = s.id
LEFT JOIN session_state_snapshots ss
  ON ss.session_id = s.id AND ss.created_at = last.max_created;

-- Skrót czatu (liczba wiadomości i ostatnia)
CREATE VIEW IF NOT EXISTS vw_session_chat_summary AS
SELECT s.id AS session_id,
       COUNT(c.id) AS messages_count,
       MAX(c.created_at) AS last_message_at
FROM sessions s
LEFT JOIN session_chats c ON c.session_id = s.id
GROUP BY s.id;

-- Ostatni pipeline_run per sesja (jeśli istnieją runy)
CREATE VIEW IF NOT EXISTS vw_session_latest_run AS
SELECT pr.*
FROM pipeline_runs pr
JOIN (
    SELECT session_id, MAX(started_at) AS max_started
    FROM pipeline_runs
    WHERE session_id IS NOT NULL
    GROUP BY session_id
) last ON last.session_id = pr.session_id
      AND last.max_started = pr.started_at;

-- ----------------------------------------------------------
-- 9) Triggery (SQLite: opcjonalnie), by aktualizować updated_at
-- (Jeśli używasz Postgresa, rozważ trigger na NOW())
-- ----------------------------------------------------------
-- sessions.updated_at
CREATE TRIGGER IF NOT EXISTS trg_sessions_updated
AFTER UPDATE ON sessions
FOR EACH ROW
BEGIN
    UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- session_metrics.updated_at
CREATE TRIGGER IF NOT EXISTS trg_smetrics_updated
AFTER UPDATE ON session_metrics
FOR EACH ROW
BEGIN
    UPDATE session_metrics SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

COMMIT;

-- ==========================================================
-- Notatki:
-- * BOOLEAN jako INTEGER 1/0 (SQLite). W Postgres można później
--   zmigrować do BOOLEAN.
-- * Pola *_json jako TEXT (w Postgres -> JSONB).
-- * Widoki ułatwiają dashboardy i szybkie zapytania.
-- * Relacja z pipelines: pipeline_runs.session_id już istnieje,
--   session_artifacts spina sesję z dowolnym artefaktem.
-- ==========================================================
