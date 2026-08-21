CREATE TABLE IF NOT EXISTS skill_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type TEXT NOT NULL,
  skill TEXT NOT NULL,
  payload TEXT NOT NULL,
  created_at_utc TEXT NOT NULL
);
