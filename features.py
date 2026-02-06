# features.py
# Extracts behavioural features from keyboard and mouse event data
# Uses sliding windows (default 10 seconds) to build feature vectors

import numpy as np
import sqlite3
from pathlib import Path


def getSessionTimeRange(conn, sessionId):
    """Get the start and end timestamps for a session."""
    cur = conn.cursor()
    cur.execute("SELECT started_at, ended_at FROM sessions WHERE id=?", (sessionId,))
    row = cur.fetchone()
    if row is None:
        return None, None
    return row[0], row[1]


def getKeyEventsInWindow(conn, sessionId, windowStart, windowEnd):
    """Grab all key events within a time window."""
    cur = conn.cursor()
    cur.execute(
        "SELECT ts, event_type, key_repr FROM key_events "
        "WHERE session_id=? AND ts>=? AND ts<?  ORDER BY ts",
        (sessionId, windowStart, windowEnd),
    )
    return cur.fetchall()


def getMouseEventsInWindow(conn, sessionId, windowStart, windowEnd):
    """Grab all mouse events within a time window."""
    cur = conn.cursor()
    cur.execute(
        "SELECT ts, event_type, dx, dy, button, scroll_dx, scroll_dy FROM mouse_events "
        "WHERE session_id=? AND ts>=? AND ts<? ORDER BY ts",
        (sessionId, windowStart, windowEnd),
    )
    return cur.fetchall()


def extractKeyboardFeatures(keyEvents):
    """
    Pull out keyboard timing features from a list of key events.
    Returns a dict of features.
    """
    features = {}

    # separate out downs and ups
    downs = [e for e in keyEvents if e[1] == "down"]
    ups = [e for e in keyEvents if e[1] == "up"]

    features["keyPressCount"] = len(downs)
    features["keyReleaseCount"] = len(ups)

    # inter-key intervals (time between consecutive key-down events)
    if len(downs) > 1:
        downTimes = [e[0] for e in downs]
        intervals = [downTimes[i+1] - downTimes[i] for i in range(len(downTimes)-1)]
        features["ikiMean"] = np.mean(intervals)
        features["ikiStd"] = np.std(intervals)
        features["ikiMin"] = np.min(intervals)
        features["ikiMax"] = np.max(intervals)
    else:
        features["ikiMean"] = 0.0
        features["ikiStd"] = 0.0
        features["ikiMin"] = 0.0
        features["ikiMax"] = 0.0

    # hold durations - match downs to ups sequentially
    # this is a rough approximation since we dont track specific keys
    holdDurations = []
    upIdx = 0
    for d in downs:
        # find the next up event after this down
        while upIdx < len(ups) and ups[upIdx][0] < d[0]:
            upIdx += 1
        if upIdx < len(ups):
            hold = ups[upIdx][0] - d[0]
            if 0 < hold < 2.0:  # filter out unreasonable holds
                holdDurations.append(hold)
            upIdx += 1

    if len(holdDurations) > 0:
        features["holdMean"] = np.mean(holdDurations)
        features["holdStd"] = np.std(holdDurations)
        features["holdMin"] = np.min(holdDurations)
        features["holdMax"] = np.max(holdDurations)
    else:
        features["holdMean"] = 0.0
        features["holdStd"] = 0.0
        features["holdMin"] = 0.0
        features["holdMax"] = 0.0

    return features


def extractMouseFeatures(mouseEvents, windowDuration=10.0):
    """
    Pull out mouse movement and interaction features.
    Returns a dict of features.
    """
    features = {}

    moves = [e for e in mouseEvents if e[1] == "move"]
    clicks = [e for e in mouseEvents if e[1] == "click"]
    scrolls = [e for e in mouseEvents if e[1] == "scroll"]

    features["moveCount"] = len(moves)
    features["clickCount"] = len(clicks)
    features["scrollCount"] = len(scrolls)

    # mouse speed and distance from move events
    if len(moves) > 0:
        distances = []
        speeds = []
        for i in range(len(moves)):
            dx = moves[i][2] or 0.0
            dy = moves[i][3] or 0.0
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(dist)

        # speed between consecutive moves
        for i in range(1, len(moves)):
            dt = moves[i][0] - moves[i-1][0]
            if dt > 0:
                dx = moves[i][2] or 0.0
                dy = moves[i][3] or 0.0
                dist = np.sqrt(dx**2 + dy**2)
                speeds.append(dist / dt)

        features["totalMouseDist"] = sum(distances)
        features["mouseDistMean"] = np.mean(distances)
        features["mouseDistStd"] = np.std(distances)

        if len(speeds) > 0:
            features["mouseSpeedMean"] = np.mean(speeds)
            features["mouseSpeedStd"] = np.std(speeds)
            features["mouseSpeedMax"] = np.max(speeds)
        else:
            features["mouseSpeedMean"] = 0.0
            features["mouseSpeedStd"] = 0.0
            features["mouseSpeedMax"] = 0.0
    else:
        features["totalMouseDist"] = 0.0
        features["mouseDistMean"] = 0.0
        features["mouseDistStd"] = 0.0
        features["mouseSpeedMean"] = 0.0
        features["mouseSpeedStd"] = 0.0
        features["mouseSpeedMax"] = 0.0

    # scroll amounts
    if len(scrolls) > 0:
        scrollAmounts = [abs(e[6] or 0.0) for e in scrolls]  # scroll_dy
        features["scrollAmountMean"] = np.mean(scrollAmounts)
    else:
        features["scrollAmountMean"] = 0.0

    return features


def computeIdleRatio(keyEvents, mouseEvents, windowDuration=10.0, idleThreshold=1.0):
    """
    Work out what fraction of the window had no activity.
    If theres a gap bigger than idleThreshold seconds between any events,
    that counts as idle time.
    """
    allTimestamps = sorted(
        [e[0] for e in keyEvents] + [e[0] for e in mouseEvents]
    )

    if len(allTimestamps) < 2:
        # basically no data, so either all idle or cant tell
        return 1.0 if len(allTimestamps) == 0 else 0.5

    idleTime = 0.0
    for i in range(1, len(allTimestamps)):
        gap = allTimestamps[i] - allTimestamps[i-1]
        if gap > idleThreshold:
            idleTime += gap

    return idleTime / windowDuration


def extractWindowFeatures(conn, sessionId, windowStart, windowEnd):
    """
    Extract the full feature vector for a single time window.
    Returns a dict with all features combined.
    """
    windowDuration = windowEnd - windowStart

    keyEvents = getKeyEventsInWindow(conn, sessionId, windowStart, windowEnd)
    mouseEvents = getMouseEventsInWindow(conn, sessionId, windowStart, windowEnd)

    # get keyboard and mouse features
    kbFeatures = extractKeyboardFeatures(keyEvents)
    mouseFeatures = extractMouseFeatures(mouseEvents, windowDuration)

    # idle ratio
    idleRatio = computeIdleRatio(keyEvents, mouseEvents, windowDuration)

    # combine everything into one dict
    combined = {}
    combined.update(kbFeatures)
    combined.update(mouseFeatures)
    combined["idleRatio"] = idleRatio
    combined["totalEvents"] = len(keyEvents) + len(mouseEvents)

    return combined


def extractSessionFeatures(conn, sessionId, windowSize=10.0, stepSize=5.0):
    """
    Run sliding window over a whole session and extract features for each window.
    Uses overlapping windows (step < window size) for more training data.
    
    Returns a list of feature dicts.
    """
    startTime, endTime = getSessionTimeRange(conn, sessionId)
    if startTime is None or endTime is None:
        print(f"Session {sessionId} has no valid time range, skipping")
        return []

    windows = []
    windowStart = startTime

    while windowStart + windowSize <= endTime:
        windowEnd = windowStart + windowSize
        feats = extractWindowFeatures(conn, sessionId, windowStart, windowEnd)
        feats["sessionId"] = sessionId
        feats["windowStart"] = windowStart
        feats["windowEnd"] = windowEnd
        windows.append(feats)
        windowStart += stepSize

    return windows


def extractAllFeatures(dbPath="auth_log.db", windowSize=10.0, stepSize=5.0):
    """
    Extract features from ALL sessions in the database.
    Returns a list of (userLabel, featureDict) tuples.
    """
    conn = sqlite3.connect(dbPath)
    cur = conn.cursor()
    cur.execute("SELECT id, user_label FROM sessions")
    sessions = cur.fetchall()

    allData = []
    for sessionId, userLabel in sessions:
        windows = extractSessionFeatures(conn, sessionId, windowSize, stepSize)
        for w in windows:
            allData.append((userLabel, w))
        print(f"  Session {sessionId} ({userLabel}): {len(windows)} windows")

    conn.close()
    return allData


# the ordered list of feature names - important to keep consistent
FEATURE_NAMES = [
    "keyPressCount", "keyReleaseCount",
    "ikiMean", "ikiStd", "ikiMin", "ikiMax",
    "holdMean", "holdStd", "holdMin", "holdMax",
    "moveCount", "clickCount", "scrollCount",
    "totalMouseDist", "mouseDistMean", "mouseDistStd",
    "mouseSpeedMean", "mouseSpeedStd", "mouseSpeedMax",
    "scrollAmountMean",
    "idleRatio", "totalEvents",
]


def featureDictToVector(featureDict):
    """Convert a feature dict into a numpy array in the right order."""
    return np.array([featureDict.get(name, 0.0) for name in FEATURE_NAMES])


# quick test if you run this file directly
if __name__ == "__main__":
    print("Extracting features from auth_log.db...")
    data = extractAllFeatures()
    print(f"\nTotal windows extracted: {len(data)}")
    if len(data) > 0:
        print(f"Feature names: {FEATURE_NAMES}")
        print(f"Example vector: {featureDictToVector(data[0][1])}")