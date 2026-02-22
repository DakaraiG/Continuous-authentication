# features.py
# Extracts behavioural features from keyboard and mouse event data
# Uses sliding windows (default 10 seconds) to build feature vectors
#
# v2 rework: 
#   - improved feature set for better discrimination
#   - typing burst detection
#   - better hold time estimation
#   - mouse trajectory features
#   - activity filtering for training

import numpy as np
import sqlite3
from pathlib import Path


# ---- database queries ----

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
        "WHERE session_id=? AND ts>=? AND ts<? ORDER BY ts",
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


# ---- keyboard features ----

def extractKeyboardFeatures(keyEvents):
    """
    Extract keyboard timing features from key events.
    Focuses on timing patterns that are unique to individuals.
    """
    features = {}

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
        features["ikiMedian"] = np.median(intervals)

        # percentiles give a better picture of the distribution shape
        features["iki25th"] = np.percentile(intervals, 25)
        features["iki75th"] = np.percentile(intervals, 75)

        # skewness and kurtosis capture whether timing is lopsided or peaky
        if len(intervals) >= 3:
            mu = np.mean(intervals)
            std = np.std(intervals)
            if std > 1e-10:
                features["ikiSkewness"] = np.mean(((intervals - mu) / std) ** 3)
                features["ikiKurtosis"] = np.mean(((intervals - mu) / std) ** 4) - 3
            else:
                features["ikiSkewness"] = 0.0
                features["ikiKurtosis"] = 0.0
        else:
            features["ikiSkewness"] = 0.0
            features["ikiKurtosis"] = 0.0
    else:
        features["ikiMean"] = 0.0
        features["ikiStd"] = 0.0
        features["ikiMin"] = 0.0
        features["ikiMax"] = 0.0
        features["ikiMedian"] = 0.0
        features["iki25th"] = 0.0
        features["iki75th"] = 0.0
        features["ikiSkewness"] = 0.0
        features["ikiKurtosis"] = 0.0

    # hold durations - match each down to the nearest following up
    holdDurations = []
    usedUps = set()

    for d in downs:
        bestUp = None
        bestDelta = float("inf")
        for uIdx, u in enumerate(ups):
            if uIdx in usedUps:
                continue
            delta = u[0] - d[0]
            # hold should be positive and reasonable (under 1.5s)
            if 0.005 < delta < 1.5 and delta < bestDelta:
                bestDelta = delta
                bestUp = uIdx
        if bestUp is not None:
            holdDurations.append(bestDelta)
            usedUps.add(bestUp)

    if len(holdDurations) > 0:
        features["holdMean"] = np.mean(holdDurations)
        features["holdStd"] = np.std(holdDurations)
        features["holdMin"] = np.min(holdDurations)
        features["holdMax"] = np.max(holdDurations)
        features["holdMedian"] = np.median(holdDurations)
        features["hold25th"] = np.percentile(holdDurations, 25)
        features["hold75th"] = np.percentile(holdDurations, 75)
    else:
        features["holdMean"] = 0.0
        features["holdStd"] = 0.0
        features["holdMin"] = 0.0
        features["holdMax"] = 0.0
        features["holdMedian"] = 0.0
        features["hold25th"] = 0.0
        features["hold75th"] = 0.0

    # typing burst analysis
    # a "burst" is a sequence of keypresses with short intervals (<0.5s)
    # the pattern of bursts vs pauses is quite individual
    burstThreshold = 0.5  # seconds - gap bigger than this means a new burst
    burstLengths = []
    burstDurations = []

    if len(downs) > 1:
        downTimes = [e[0] for e in downs]
        currentBurstStart = downTimes[0]
        currentBurstCount = 1

        for i in range(1, len(downTimes)):
            gap = downTimes[i] - downTimes[i-1]
            if gap < burstThreshold:
                currentBurstCount += 1
            else:
                # end of a burst
                burstLengths.append(currentBurstCount)
                burstDurations.append(downTimes[i-1] - currentBurstStart)
                currentBurstStart = downTimes[i]
                currentBurstCount = 1

        # dont forget the last burst
        burstLengths.append(currentBurstCount)
        burstDurations.append(downTimes[-1] - currentBurstStart)

    if len(burstLengths) > 0:
        features["burstCount"] = len(burstLengths)
        features["burstLengthMean"] = np.mean(burstLengths)
        features["burstLengthStd"] = np.std(burstLengths) if len(burstLengths) > 1 else 0.0
        features["burstDurationMean"] = np.mean(burstDurations)
    else:
        features["burstCount"] = 0
        features["burstLengthMean"] = 0.0
        features["burstLengthStd"] = 0.0
        features["burstDurationMean"] = 0.0

    # typing rate (keys per second in active periods)
    if len(downs) >= 2:
        downTimes = [e[0] for e in downs]
        activeTime = downTimes[-1] - downTimes[0]
        if activeTime > 0:
            features["typingRate"] = len(downs) / activeTime
        else:
            features["typingRate"] = 0.0
    else:
        features["typingRate"] = 0.0

    # key overlap ratio - fraction of keypresses where the next key is pressed
    # before the previous key is released. fast typists overlap a lot, slow typists don't.
    # detected purely from timing order of down/up events, no key identity needed.
    if len(downs) > 0:
        allKeyEvents = sorted(
            [(e[0], "down") for e in downs] + [(e[0], "up") for e in ups]
        )
        heldCount = 0
        overlapCount = 0
        for _, evType in allKeyEvents:
            if evType == "down":
                if heldCount > 0:
                    overlapCount += 1
                heldCount += 1
            else:
                heldCount = max(0, heldCount - 1)
        features["overlapRatio"] = overlapCount / len(downs)
    else:
        features["overlapRatio"] = 0.0

    return features


# ---- mouse features ----

def extractMouseFeatures(mouseEvents, windowDuration=10.0):
    """
    Extract mouse movement and interaction features.
    Includes trajectory analysis for more individual discrimination.
    """
    features = {}

    moves = [e for e in mouseEvents if e[1] == "move"]
    clicks = [e for e in mouseEvents if e[1] == "click"]
    scrolls = [e for e in mouseEvents if e[1] == "scroll"]

    features["moveCount"] = len(moves)
    features["clickCount"] = len(clicks)
    features["scrollCount"] = len(scrolls)

    if len(moves) > 0:
        distances = []
        speeds = []
        angles = []
        accelerations = []

        for i in range(len(moves)):
            dx = moves[i][2] or 0.0
            dy = moves[i][3] or 0.0
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(dist)

            # movement angle
            if dist > 0.01:  # avoid division by zero on tiny moves
                angle = np.arctan2(dy, dx)
                angles.append(angle)

        # speed between consecutive move events
        for i in range(1, len(moves)):
            dt = moves[i][0] - moves[i-1][0]
            if dt > 0:
                dx = moves[i][2] or 0.0
                dy = moves[i][3] or 0.0
                dist = np.sqrt(dx**2 + dy**2)
                speed = dist / dt
                speeds.append(speed)

        # acceleration (change in speed)
        for i in range(1, len(speeds)):
            accelerations.append(abs(speeds[i] - speeds[i-1]))

        features["totalMouseDist"] = sum(distances)
        features["mouseDistMean"] = np.mean(distances)
        features["mouseDistStd"] = np.std(distances)

        # path efficiency - ratio of net displacement to total path length
        # 1.0 = perfectly straight line, lower = curved/wandering movement
        netDx = sum(m[2] or 0.0 for m in moves)
        netDy = sum(m[3] or 0.0 for m in moves)
        netDisplacement = np.sqrt(netDx**2 + netDy**2)
        totalDist = features["totalMouseDist"]
        features["pathEfficiency"] = netDisplacement / totalDist if totalDist > 0 else 0.0

        if len(speeds) > 0:
            features["mouseSpeedMean"] = np.mean(speeds)
            features["mouseSpeedStd"] = np.std(speeds)
            features["mouseSpeedMax"] = np.max(speeds)
            features["mouseSpeedMedian"] = np.median(speeds)
        else:
            features["mouseSpeedMean"] = 0.0
            features["mouseSpeedStd"] = 0.0
            features["mouseSpeedMax"] = 0.0
            features["mouseSpeedMedian"] = 0.0

        # acceleration features
        if len(accelerations) > 0:
            features["mouseAccelMean"] = np.mean(accelerations)
            features["mouseAccelStd"] = np.std(accelerations)
        else:
            features["mouseAccelMean"] = 0.0
            features["mouseAccelStd"] = 0.0

        # direction change frequency - how often the mouse changes direction
        # this is quite individual (some people move in smooth arcs, others jitter)
        if len(angles) > 2:
            directionChanges = 0
            for i in range(1, len(angles)):
                # angle difference, wrapped to [-pi, pi]
                diff = angles[i] - angles[i-1]
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                if abs(diff) > np.pi / 4:  # more than 45 degrees counts as a change
                    directionChanges += 1
            features["directionChangeRate"] = directionChanges / len(angles)
        else:
            features["directionChangeRate"] = 0.0

    else:
        features["totalMouseDist"] = 0.0
        features["mouseDistMean"] = 0.0
        features["mouseDistStd"] = 0.0
        features["pathEfficiency"] = 0.0
        features["mouseSpeedMean"] = 0.0
        features["mouseSpeedStd"] = 0.0
        features["mouseSpeedMax"] = 0.0
        features["mouseSpeedMedian"] = 0.0
        features["mouseAccelMean"] = 0.0
        features["mouseAccelStd"] = 0.0
        features["directionChangeRate"] = 0.0

    # scroll behaviour
    if len(scrolls) > 0:
        scrollAmounts = [abs(e[6] or 0.0) for e in scrolls]
        features["scrollAmountMean"] = np.mean(scrollAmounts)
        features["scrollAmountStd"] = np.std(scrollAmounts)

        # scroll timing
        if len(scrolls) > 1:
            scrollTimes = [e[0] for e in scrolls]
            scrollIntervals = [scrollTimes[i+1] - scrollTimes[i] for i in range(len(scrollTimes)-1)]
            features["scrollIntervalMean"] = np.mean(scrollIntervals)
        else:
            features["scrollIntervalMean"] = 0.0
    else:
        features["scrollAmountMean"] = 0.0
        features["scrollAmountStd"] = 0.0
        features["scrollIntervalMean"] = 0.0

    # click timing
    if len(clicks) > 1:
        clickTimes = [e[0] for e in clicks]
        clickIntervals = [clickTimes[i+1] - clickTimes[i] for i in range(len(clickTimes)-1)]
        features["clickIntervalMean"] = np.mean(clickIntervals)
        features["clickIntervalStd"] = np.std(clickIntervals)
    else:
        features["clickIntervalMean"] = 0.0
        features["clickIntervalStd"] = 0.0

    # left vs right click ratio - some people are right-click-heavy, others rarely right-click
    if len(clicks) > 0:
        leftClicks = sum(1 for e in clicks if "left" in str(e[4]).lower())
        features["leftClickRatio"] = leftClicks / len(clicks)
    else:
        features["leftClickRatio"] = 0.5  # neutral default

    # pre-click deceleration - do they slow down as they approach a click target?
    # compare mean speed in the 0.25s before a click vs the 0.25s before that
    decelerations = []
    for clickTs in [e[0] for e in clicks]:
        preMoves = [m for m in moves if clickTs - 0.5 <= m[0] < clickTs]
        if len(preMoves) < 4:
            continue
        preSpeeds = []
        for i in range(1, len(preMoves)):
            dt = preMoves[i][0] - preMoves[i-1][0]
            if dt > 0:
                dx = preMoves[i][2] or 0.0
                dy = preMoves[i][3] or 0.0
                preSpeeds.append(np.sqrt(dx**2 + dy**2) / dt)
        if len(preSpeeds) >= 2:
            mid = len(preSpeeds) // 2
            earlySpeed = np.mean(preSpeeds[:mid])
            lateSpeed = np.mean(preSpeeds[mid:])
            if earlySpeed > 0:
                decelerations.append((earlySpeed - lateSpeed) / earlySpeed)
    features["preClickDecel"] = np.mean(decelerations) if decelerations else 0.0

    return features


# ---- combined features ----

def computeIdleRatio(keyEvents, mouseEvents, windowDuration=10.0, idleThreshold=1.0):
    """
    Work out what fraction of the window had no activity.
    Gaps bigger than idleThreshold count as idle time.
    """
    allTimestamps = sorted(
        [e[0] for e in keyEvents] + [e[0] for e in mouseEvents]
    )

    if len(allTimestamps) < 2:
        return 1.0 if len(allTimestamps) == 0 else 0.5

    idleTime = 0.0
    for i in range(1, len(allTimestamps)):
        gap = allTimestamps[i] - allTimestamps[i-1]
        if gap > idleThreshold:
            idleTime += gap

    return min(idleTime / windowDuration, 1.0)


def computeInteractionRatio(keyEvents, mouseEvents):
    """
    Ratio of keyboard to mouse events.
    Some people are keyboard-heavy, others are mouse-heavy.
    """
    kCount = len(keyEvents)
    mCount = len(mouseEvents)
    total = kCount + mCount
    if total == 0:
        return 0.5  # neutral
    return kCount / total


def extractWindowFeatures(conn, sessionId, windowStart, windowEnd):
    """
    Extract the full feature vector for a single time window.
    Returns a dict with all features.
    """
    windowDuration = windowEnd - windowStart

    keyEvents = getKeyEventsInWindow(conn, sessionId, windowStart, windowEnd)
    mouseEvents = getMouseEventsInWindow(conn, sessionId, windowStart, windowEnd)

    kbFeatures = extractKeyboardFeatures(keyEvents)
    mouseFeatures = extractMouseFeatures(mouseEvents, windowDuration)

    idleRatio = computeIdleRatio(keyEvents, mouseEvents, windowDuration)
    interactionRatio = computeInteractionRatio(keyEvents, mouseEvents)

    combined = {}
    combined.update(kbFeatures)
    combined.update(mouseFeatures)
    combined["idleRatio"] = idleRatio
    combined["kbMouseRatio"] = interactionRatio
    combined["totalEvents"] = len(keyEvents) + len(mouseEvents)

    return combined


def extractSessionFeatures(conn, sessionId, windowSize=10.0, stepSize=5.0, minEvents=10):
    """
    Sliding window over a session, extracting features per window.
    
    minEvents: skip windows with fewer than this many events.
    This filters out idle periods that would just add noise to the model.
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

        # only keep windows with enough activity to be meaningful
        if feats.get("totalEvents", 0) >= minEvents:
            feats["sessionId"] = sessionId
            feats["windowStart"] = windowStart
            feats["windowEnd"] = windowEnd
            windows.append(feats)

        windowStart += stepSize

    return windows


def extractAllFeatures(dbPath="auth_log.db", windowSize=10.0, stepSize=5.0, minEvents=10):
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
        windows = extractSessionFeatures(conn, sessionId, windowSize, stepSize, minEvents)
        for w in windows:
            allData.append((userLabel, w))
        print(f"  Session {sessionId} ({userLabel}): {len(windows)} windows")

    conn.close()
    return allData


# the ordered list of feature names - must stay consistent between training and inference
FEATURE_NAMES = [
    # keyboard timing
    "keyPressCount", "keyReleaseCount",
    "ikiMean", "ikiStd", "ikiMin", "ikiMax", "ikiMedian", "iki25th", "iki75th",
    "ikiSkewness", "ikiKurtosis",
    "holdMean", "holdStd", "holdMin", "holdMax", "holdMedian", "hold25th", "hold75th",
    # typing patterns
    "burstCount", "burstLengthMean", "burstLengthStd", "burstDurationMean",
    "typingRate", "overlapRatio",
    # mouse movement
    "moveCount", "clickCount", "scrollCount",
    "totalMouseDist", "mouseDistMean", "mouseDistStd", "pathEfficiency",
    "mouseSpeedMean", "mouseSpeedStd", "mouseSpeedMax", "mouseSpeedMedian",
    "mouseAccelMean", "mouseAccelStd",
    "directionChangeRate",
    # mouse interaction
    "scrollAmountMean", "scrollAmountStd", "scrollIntervalMean",
    "clickIntervalMean", "clickIntervalStd",
    "leftClickRatio", "preClickDecel",
    # combined
    "idleRatio", "kbMouseRatio", "totalEvents",
]


def featureDictToVector(featureDict):
    """Convert a feature dict into a numpy array in the right order."""
    return np.array([featureDict.get(name, 0.0) for name in FEATURE_NAMES])


if __name__ == "__main__":
    print("Extracting features from auth_log.db...")
    data = extractAllFeatures()
    print(f"\nTotal windows extracted: {len(data)}")
    if len(data) > 0:
        print(f"Number of features: {len(FEATURE_NAMES)}")
        print(f"Feature names: {FEATURE_NAMES}")
        vec = featureDictToVector(data[0][1])
        print(f"Example vector: {vec}")