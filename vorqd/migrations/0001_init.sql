-- In-flight backend handles: one row per job whose async work is running.
-- `IF NOT EXISTS` adopts state files written before migrations existed.
CREATE TABLE IF NOT EXISTS inflight (
  job_id TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  handle TEXT NOT NULL,
  created_at REAL NOT NULL
);
