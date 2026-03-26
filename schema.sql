-- MARC2 Schema
-- Extended from marc-from-larc with solve_trials and descriptions tables

-- Tasks from ARC-AGI2
CREATE TABLE IF NOT EXISTS tasks (
    task_id          INTEGER PRIMARY KEY,  -- training 0-999, evaluation 2000-2119
    arc_name         TEXT NOT NULL UNIQUE,  -- original filename (hex ID)
    num_train        INTEGER NOT NULL,
    source           TEXT NOT NULL DEFAULT 'training'  -- 'training' or 'evaluation'
);

-- Phase 1: Claude's solving attempts (via Claude Code subagents)
CREATE TABLE IF NOT EXISTS solve_trials (
    trial_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id       INTEGER NOT NULL REFERENCES tasks(task_id),
    attempt       INTEGER NOT NULL DEFAULT 1,  -- 1 or 2
    prompt_text   TEXT,
    reasoning     TEXT,                    -- full reasoning trace (the gold)
    predicted_grid TEXT,                   -- JSON 2D array
    correct       INTEGER,                -- 1/0/NULL
    cell_accuracy REAL,
    error         TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(task_id, attempt)
);

-- Phase 2: Distilled language-complete descriptions
CREATE TABLE IF NOT EXISTS descriptions (
    desc_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id          INTEGER NOT NULL REFERENCES tasks(task_id) UNIQUE,
    source_trial_id  INTEGER REFERENCES solve_trials(trial_id),
    see_description  TEXT NOT NULL,
    do_description   TEXT NOT NULL,
    grid_description TEXT NOT NULL,
    validated        INTEGER DEFAULT 0,      -- 0=untested, 1=passed, -1=failed
    validation_model TEXT,
    validation_correct INTEGER,
    revision_count   INTEGER DEFAULT 0,
    generation_model TEXT NOT NULL DEFAULT 'claude-subagent',
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Phase 4: Subject model baseline trials (inherited from marc-from-larc)
CREATE TABLE IF NOT EXISTS baseline_trials (
    trial_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id        INTEGER NOT NULL REFERENCES tasks(task_id),
    model_name     TEXT NOT NULL,
    condition      TEXT NOT NULL,           -- 'examples_only','language_only','both'
    num_examples   INTEGER NOT NULL,
    repeat_num     INTEGER NOT NULL DEFAULT 1,
    prompt_text    TEXT NOT NULL,
    raw_response   TEXT,
    response_text  TEXT,
    response_at    TEXT,
    latency_ms     INTEGER,
    error          TEXT,
    predicted_grid TEXT,                    -- JSON 2D array
    reasoning      TEXT,
    correct        INTEGER,
    cell_accuracy  REAL,
    UNIQUE(task_id, model_name, condition, num_examples, repeat_num)
);

-- Phase 5: Task classification (inherited)
CREATE TABLE IF NOT EXISTS task_subsets (
    task_id                    INTEGER NOT NULL REFERENCES tasks(task_id),
    model_name                 TEXT NOT NULL,
    subset                     TEXT NOT NULL,
    min_examples_with_language INTEGER,
    PRIMARY KEY (task_id, model_name)
);

-- Phase 6-7: Figurative descriptions (inherited)
CREATE TABLE IF NOT EXISTS figurative_descriptions (
    fig_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id          INTEGER NOT NULL REFERENCES tasks(task_id),
    generator_model  TEXT NOT NULL,
    variant          TEXT NOT NULL DEFAULT 'original',
    source_domain    TEXT,
    metaphor         TEXT NOT NULL,
    figurative_see   TEXT NOT NULL,
    figurative_do    TEXT NOT NULL,
    figurative_grid  TEXT NOT NULL,
    generation_prompt TEXT,
    raw_generation   TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(task_id, generator_model, variant)
);

-- Phase 7: Figurative trials (inherited)
CREATE TABLE IF NOT EXISTS figurative_trials (
    trial_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    fig_id         INTEGER NOT NULL REFERENCES figurative_descriptions(fig_id),
    task_id        INTEGER NOT NULL REFERENCES tasks(task_id),
    model_name     TEXT NOT NULL,
    num_examples   INTEGER NOT NULL,       -- 0 = figurative only
    repeat_num     INTEGER NOT NULL DEFAULT 1,
    prompt_text    TEXT NOT NULL,
    raw_response   TEXT,
    response_text  TEXT,
    response_at    TEXT,
    latency_ms     INTEGER,
    error          TEXT,
    predicted_grid TEXT,
    reasoning      TEXT,
    correct        INTEGER,
    cell_accuracy  REAL,
    UNIQUE(fig_id, model_name, num_examples, repeat_num)
);
