-- ==========================================================
-- DataGenius PRO - Model Registry Schema (+ seed)
-- Portable SQL for SQLite / PostgreSQL
-- ==========================================================

PRAGMA foreign_keys = ON;          -- SQLite (ignorowane w Postgres)
BEGIN;

-- ----------------------------------------------------------
-- 1) Tabela: model_registry
--    Katalog dostępnych algorytmów dla danego typu problemu
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_registry (
    id               TEXT PRIMARY KEY,                     -- UUID/slug (np. 'clf_xgboost')
    code             TEXT NOT NULL,                        -- krótka nazwa: lr, rf, xgboost
    problem_type     TEXT NOT NULL,                        -- classification|regression
    name             TEXT NOT NULL,                        -- pełna nazwa
    category         TEXT,                                 -- linear|tree_based|ensemble|boosting|...
    description      TEXT,
    pros_json        TEXT,                                 -- JSON: ["Fast","Interpretable",...]
    cons_json        TEXT,                                 -- JSON: ["May underfit",...]
    best_for_json    TEXT,                                 -- JSON: ["Tabular data",...]
    default_params   TEXT,                                 -- JSON domyślnych hiperparametrów
    turbo            INTEGER NOT NULL DEFAULT 1,           -- 1/0 (bool)
    enabled          INTEGER NOT NULL DEFAULT 1,           -- 1/0 (bool)
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(code, problem_type)
);

CREATE INDEX IF NOT EXISTS idx_registry_problem ON model_registry(problem_type);
CREATE INDEX IF NOT EXISTS idx_registry_enabled ON model_registry(enabled);

-- ----------------------------------------------------------
-- 2) Tabela: model_selection_strategies
--    Strategie wyboru modeli (fast/accurate/interpretable)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_selection_strategies (
    id               TEXT PRIMARY KEY,                     -- np. 'accurate'
    description      TEXT NOT NULL,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------
-- 3) Tabela: strategy_models (relacja N:M)
--    Mapowanie strategii -> listy modeli w danym problem_type
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS strategy_models (
    strategy_id      TEXT NOT NULL REFERENCES model_selection_strategies(id) ON DELETE CASCADE,
    registry_id      TEXT NOT NULL REFERENCES model_registry(id) ON DELETE CASCADE,
    rank_order       INTEGER NOT NULL DEFAULT 0,           -- kolejność w strategii
    PRIMARY KEY (strategy_id, registry_id)
);

CREATE INDEX IF NOT EXISTS idx_strategy_models_order ON strategy_models(strategy_id, rank_order);

-- ----------------------------------------------------------
-- 4) Widok pomocniczy: tylko aktywne modele per problem_type
-- ----------------------------------------------------------
CREATE VIEW IF NOT EXISTS vw_registry_enabled_by_problem AS
SELECT
    problem_type,
    code,
    name,
    category,
    turbo,
    id AS registry_id
FROM model_registry
WHERE enabled = 1
ORDER BY problem_type, code;

-- ==========================================================
-- SEED (odwzorowanie config/model_registry.py)
-- Uwaga: dla prostoty używamy stałych ID w stylu '<ptype>_<code>'
-- ==========================================================

-- Strategie
INSERT OR IGNORE INTO model_selection_strategies (id, description) VALUES
('fast',         'Quick models for rapid prototyping'),
('accurate',     'Focus on accuracy (slower training)'),
('interpretable','Interpretable models for explanations');

-- ---------- Classification ----------
INSERT OR IGNORE INTO model_registry
(id, code, problem_type, name, category, description, pros_json, cons_json, best_for_json, default_params, turbo, enabled)
VALUES
('classification_lr','lr','classification','Logistic Regression','linear',
 'Linear model for binary/multiclass classification',
 '["Fast","Interpretable","Good baseline"]',
 '["Assumes linear relationship","May underfit"]',
 '["Linear separable data","High-dimensional data"]',
 '{}',1,1),

('classification_knn','knn','classification','K-Nearest Neighbors','linear',
 'Instance-based learning algorithm',
 '["Simple","No training phase"]',
 '["Slow predictions","Sensitive to scale"]',
 '["Small datasets","Low dimensions"]',
 '{}',0,1),

('classification_nb','nb','classification','Naive Bayes','bayesian',
 'Probabilistic classifier based on Bayes theorem',
 '["Fast","Works well with small data"]',
 '["Assumes feature independence"]',
 '["Text classification","Small datasets"]',
 '{}',1,1),

('classification_dt','dt','classification','Decision Tree','tree_based',
 'Tree-based classifier',
 '["Interpretable","Handles non-linear"]',
 '["Prone to overfitting"]',
 '["Categorical features","Rule extraction"]',
 '{}',1,1),

('classification_svm','svm','classification','Support Vector Machine','linear',
 'Maximum margin classifier',
 '["Effective in high dimensions","Memory efficient"]',
 '["Slow on large datasets","Hard to interpret"]',
 '["High-dimensional data","Clear margin"]',
 '{}',0,1),

('classification_rf','rf','classification','Random Forest','ensemble',
 'Ensemble of decision trees',
 '["Robust","Handles missing values","Feature importance"]',
 '["Memory intensive","Slow inference"]',
 '["Tabular data","Feature importance"]',
 '{}',1,1),

('classification_et','et','classification','Extra Trees','ensemble',
 'Extremely randomized trees',
 '["Faster than RF","Less overfitting"]',
 '["May underfit"]',
 '["Similar to Random Forest"]',
 '{}',1,1),

('classification_gbc','gbc','classification','Gradient Boosting','boosting',
 'Sequential ensemble method',
 '["High accuracy","Handles various data types"]',
 '["Slow training","Prone to overfitting"]',
 '["Tabular data","Competitions"]',
 '{}',1,1),

('classification_xgboost','xgboost','classification','XGBoost','boosting',
 'Optimized gradient boosting',
 '["State-of-the-art","Fast","Regularization"]',
 '["Complex hyperparameters"]',
 '["Structured data","Competitions"]',
 '{}',1,1),

('classification_lightgbm','lightgbm','classification','LightGBM','boosting',
 'Fast gradient boosting framework',
 '["Very fast","Memory efficient","Handles large data"]',
 '["May overfit small data"]',
 '["Large datasets","Fast training"]',
 '{}',1,1),

('classification_catboost','catboost','classification','CatBoost','boosting',
 'Gradient boosting with categorical support',
 '["Handles categorical features","Less hyperparameters"]',
 '["Slower than LightGBM"]',
 '["Categorical features","Less tuning needed"]',
 '{}',1,1),

('classification_ada','ada','classification','AdaBoost','boosting',
 'Adaptive boosting algorithm',
 '["Simple","Less prone to overfitting"]',
 '["Sensitive to outliers"]',
 '["Binary classification","Weak learners"]',
 '{}',1,1);

-- ---------- Regression ----------
INSERT OR IGNORE INTO model_registry
(id, code, problem_type, name, category, description, pros_json, cons_json, best_for_json, default_params, turbo, enabled)
VALUES
('regression_lr','lr','regression','Linear Regression','linear',
 'Ordinary least squares regression',
 '["Simple","Interpretable","Fast"]',
 '["Assumes linearity"]',
 '["Linear relationships","Baseline"]',
 '{}',1,1),

('regression_ridge','ridge','regression','Ridge Regression','linear',
 'L2 regularized linear regression',
 '["Handles multicollinearity","Stable"]',
 '["May underfit"]',
 '["Correlated features"]',
 '{}',1,1),

('regression_lasso','lasso','regression','Lasso Regression','linear',
 'L1 regularized linear regression',
 '["Feature selection","Sparse models"]',
 '["May underfit"]',
 '["High-dimensional data","Feature selection"]',
 '{}',1,1),

('regression_en','en','regression','Elastic Net','linear',
 'L1 + L2 regularized regression',
 '["Combines Ridge and Lasso","Robust"]',
 '["More hyperparameters"]',
 '["Correlated features + sparsity"]',
 '{}',1,1),

('regression_dt','dt','regression','Decision Tree','tree_based',
 'Tree-based regressor',
 '["Interpretable","Handles non-linear"]',
 '["Prone to overfitting"]',
 '["Non-linear relationships"]',
 '{}',1,1),

('regression_rf','rf','regression','Random Forest','ensemble',
 'Ensemble of decision trees',
 '["Robust","Handles outliers","Feature importance"]',
 '["Memory intensive"]',
 '["Tabular data","Non-linear"]',
 '{}',1,1),

('regression_et','et','regression','Extra Trees','ensemble',
 'Extremely randomized trees',
 '["Faster than RF","Less overfitting"]',
 '["May underfit"]',
 '["Similar to Random Forest"]',
 '{}',1,1),

('regression_gbr','gbr','regression','Gradient Boosting','boosting',
 'Sequential ensemble for regression',
 '["High accuracy","Flexible"]',
 '["Slow training"]',
 '["Complex relationships"]',
 '{}',1,1),

('regression_xgboost','xgboost','regression','XGBoost','boosting',
 'Optimized gradient boosting',
 '["State-of-the-art","Fast","Regularization"]',
 '["Complex hyperparameters"]',
 '["Structured data","Competitions"]',
 '{}',1,1),

('regression_lightgbm','lightgbm','regression','LightGBM','boosting',
 'Fast gradient boosting',
 '["Very fast","Memory efficient"]',
 '["May overfit small data"]',
 '["Large datasets"]',
 '{}',1,1),

('regression_catboost','catboost','regression','CatBoost','boosting',
 'Gradient boosting with categorical support',
 '["Handles categorical","Less tuning"]',
 '["Slower than LightGBM"]',
 '["Categorical features"]',
 '{}',1,1);

-- ----------------------------------------------------------
-- Mapowanie strategii -> modele (kolejność jak w configu)
-- FAST
INSERT OR IGNORE INTO strategy_models (strategy_id, registry_id, rank_order) VALUES
('fast','classification_lr',1), ('fast','classification_dt',2), ('fast','classification_rf',3),
('fast','regression_lr',1),    ('fast','regression_ridge',2),  ('fast','regression_rf',3);

-- ACCURATE
INSERT OR IGNORE INTO strategy_models (strategy_id, registry_id, rank_order) VALUES
('accurate','classification_xgboost',1),
('accurate','classification_lightgbm',2),
('accurate','classification_catboost',3),
('accurate','classification_rf',4),
('accurate','classification_et',5),

('accurate','regression_xgboost',1),
('accurate','regression_lightgbm',2),
('accurate','regression_catboost',3),
('accurate','regression_rf',4),
('accurate','regression_et',5);

-- INTERPRETABLE
INSERT OR IGNORE INTO strategy_models (strategy_id, registry_id, rank_order) VALUES
('interpretable','classification_lr',1), ('interpretable','classification_dt',2),
('interpretable','regression_lr',1),    ('interpretable','regression_ridge',2), ('interpretable','regression_dt',3);

COMMIT;

-- ==========================================================
-- Notatki:
-- * JSON trzymamy w TEXT (zgodne z SQLite). W Postgres można zmigrować do JSONB.
-- * Pola turbo/enabled: 1/0 (SQLite). W Postgres można zrobić ALTER na BOOLEAN.
-- * Jeśli używasz Alembica, załaduj seed przez migration/script lub jednorazowo.
-- ==========================================================
