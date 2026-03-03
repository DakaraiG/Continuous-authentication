# logger.py
# Automatic CSV logger for external testing
# Writes inference results, session markers and alert events to a CSV file that presists across runs.


import csv
import os
import time
from pathlib import Path

LOG_FILENAME = "session_log.csv"

COLUMNS = [
    "timestamp",          # unix timestamp
    "datetime",           # human-readable
    "event_type",         # session_start | inference | alert_triggered | alert_confirmed | alert_dismissed | session_end
    "session_id",
    "user_label",
    "confidence",         # 0-100 scale, blank for session_start/end
    "policy_state",       # accepted | warning | suspicious
    "total_events",       # events in the 10s window
    "key_events",         # keyboard events in window
    "mouse_events",       # mouse events in window
    "consecutive_low",    # how many suspicious windows in a row
    "accept_threshold",   # current threshold setting
    "challenge_threshold",
    "session_duration",   # only filled on session_end rows
    "notes",              # free text for anything extra
]


class SessionLogger:
    """
    Appends rows to a CSV file automatically.
    Creates the file with headers if it doesnt exist yet,
    otherwise just keeps appending.
    """
    def __init__(self, logDir=None):
        if logDir:
            self.logPath = Path(logDir) / LOG_FILENAME
        else:
            self.logPath = Path(LOG_FILENAME)

        self._ensureFile()

    def _ensureFile(self):
        """Create the CSV with headers if it doesnt exist."""
        if not self.logPath.exists():
            with open(self.logPath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(COLUMNS)

    def _writeRow(self, data):
        """Append a single row. data is a dict matching COLUMNS."""
        with open(self.logPath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            row = [data.get(col, "") for col in COLUMNS]
            writer.writerow(row)

    def logSessionStart(self, sessionId, userLabel):
        """Log when a capture session begins."""
        now = time.time()
        self._writeRow({
            "timestamp": now,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "event_type": "session_start",
            "session_id": sessionId,
            "user_label": userLabel,
            "notes": "Session started",
        })

    def logInference(self, sessionId, userLabel, confidence, policyState,
                     totalEvents, keyEvents, mouseEvents, consecutiveLow,
                     acceptThreshold, challengeThreshold):
        """Log a single inference window result."""
        now = time.time()
        self._writeRow({
            "timestamp": now,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "event_type": "inference",
            "session_id": sessionId,
            "user_label": userLabel,
            "confidence": f"{confidence:.2f}",
            "policy_state": policyState,
            "total_events": totalEvents,
            "key_events": keyEvents,
            "mouse_events": mouseEvents,
            "consecutive_low": consecutiveLow,
            "accept_threshold": acceptThreshold,
            "challenge_threshold": challengeThreshold,
        })

    def logAlert(self, sessionId, userLabel, confidence, eventType, consecutiveLow):
        """
        Log an alert event.
        eventType should be: alert_triggered, alert_confirmed, or alert_dismissed
        """
        now = time.time()
        self._writeRow({
            "timestamp": now,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "event_type": eventType,
            "session_id": sessionId,
            "user_label": userLabel,
            "confidence": f"{confidence:.2f}",
            "consecutive_low": consecutiveLow,
        })

    def logSessionEnd(self, sessionId, userLabel, duration):
        """Log when a capture session ends."""
        now = time.time()
        self._writeRow({
            "timestamp": now,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "event_type": "session_end",
            "session_id": sessionId,
            "user_label": userLabel,
            "session_duration": f"{duration:.1f}",
            "notes": "Session ended",
        })