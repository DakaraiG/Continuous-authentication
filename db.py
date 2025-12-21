# db.py
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

dbName = "auth_log.db"

schemaSql = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_label TEXT NOT NULL,
  mode TEXT NOT NULL,
  started_at REAL NOT NULL,
  ended_at REAL
);

CREATE TABLE IF NOT EXISTS key_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL,
  ts REAL NOT NULL,
  event_type TEXT NOT NULL,       -- "down" | "up"
  key_repr TEXT,                  -- privacy-safe (can be NULL)
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS mouse_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL,
  ts REAL NOT NULL,
  event_type TEXT NOT NULL,       -- "move" | "click" | "scroll"
  dx REAL,
  dy REAL,
  button TEXT,
  scroll_dx REAL,
  scroll_dy REAL,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_key_session_ts
  ON key_events(session_id, ts);

CREATE INDEX IF NOT EXISTS idx_mouse_session_ts
  ON mouse_events(session_id, ts);
"""

class Db:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or dbName)
        self.conn = sqlite3.connect(self.path.as_posix(), check_same_thread=False)
        self.conn.executescript(schemaSql)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def startSession(self, userLabel: str, mode: str, startedAt: float) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO sessions(user_label, mode, started_at) VALUES (?, ?, ?)",
            (userLabel, mode, startedAt),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def endSession(self, sessionId: int, endedAt: float):
        self.conn.execute(
            "UPDATE sessions SET ended_at=? WHERE id=?",
            (endedAt, sessionId),
        )
        self.conn.commit()

    def insertKeyEvent(self, sessionId: int, ts: float, eventType: str, keyRepr: Optional[str]):
        self.conn.execute(
            "INSERT INTO key_events(session_id, ts, event_type, key_repr) VALUES (?, ?, ?, ?)",
            (sessionId, ts, eventType, keyRepr),
        )

    def insertMouseMove(self, sessionId: int, ts: float, dx: float, dy: float):
        self.conn.execute(
            "INSERT INTO mouse_events(session_id, ts, event_type, dx, dy) VALUES (?, ?, 'move', ?, ?)",
            (sessionId, ts, dx, dy),
        )

    def insertMouseClick(self, sessionId: int, ts: float, button: str):
        self.conn.execute(
            "INSERT INTO mouse_events(session_id, ts, event_type, button) VALUES (?, ?, 'click', ?)",
            (sessionId, ts, button),
        )

    def insertMouseScroll(self, sessionId: int, ts: float, scrollDx: float, scrollDy: float):
        self.conn.execute(
            "INSERT INTO mouse_events(session_id, ts, event_type, scroll_dx, scroll_dy) VALUES (?, ?, 'scroll', ?, ?)",
            (sessionId, ts, scrollDx, scrollDy),
        )

    def commit(self):
        self.conn.commit()

    def getEventCountsForSession(self, sessionId: int) -> Tuple[int, int]:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM key_events WHERE session_id=?", (sessionId,))
        keyCount = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM mouse_events WHERE session_id=?", (sessionId,))
        mouseCount = cur.fetchone()[0]
        return int(keyCount), int(mouseCount)
