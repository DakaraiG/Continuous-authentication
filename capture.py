# capture.py
import time
from dataclasses import dataclass
from typing import Optional

from pynput import keyboard, mouse

@dataclass
class CaptureConfig:
    privacyMode: bool = True
    mouseMoveSampleHz: float = 50.0

class GlobalCapture:
    """
    System-wide capture using pynput (Windows + macOS).
    """
    def __init__(self, db, sessionId: int, cfg: Optional[CaptureConfig] = None):
        self.db = db
        self.sessionId = sessionId
        self.cfg = cfg or CaptureConfig()

        self.kbListener = None
        self.mouseListener = None
        self.running = False

        self.lastMousePos = None
        self.lastMouseEmit = 0.0

    def safeKeyRepr(self, key) -> str:
        if self.cfg.privacyMode:
            return "<key>"
        try:
            return str(key)
        except Exception:
            return "<key>"

    def onKeyPress(self, key):
        ts = time.time()
        self.db.insertKeyEvent(self.sessionId, ts, "down", self.safeKeyRepr(key))

    def onKeyRelease(self, key):
        ts = time.time()
        self.db.insertKeyEvent(self.sessionId, ts, "up", self.safeKeyRepr(key))

    def onMove(self, x, y):
        ts = time.time()
        if ts - self.lastMouseEmit < (1.0 / self.cfg.mouseMoveSampleHz):
            return
        self.lastMouseEmit = ts

        if self.lastMousePos is None:
            self.lastMousePos = (x, y)
            return

        lx, ly = self.lastMousePos
        dx, dy = float(x - lx), float(y - ly)
        self.lastMousePos = (x, y)
        self.db.insertMouseMove(self.sessionId, ts, dx, dy)

    def onClick(self, x, y, button, pressed):
        if not pressed:
            return
        ts = time.time()
        self.db.insertMouseClick(self.sessionId, ts, str(button))

    def onScroll(self, x, y, dx, dy):
        ts = time.time()
        self.db.insertMouseScroll(self.sessionId, ts, float(dx), float(dy))

    def start(self):
        if self.running:
            return
        self.running = True
        self.kbListener = keyboard.Listener(on_press=self.onKeyPress, on_release=self.onKeyRelease)
        self.mouseListener = mouse.Listener(on_move=self.onMove, on_click=self.onClick, on_scroll=self.onScroll)
        self.kbListener.start()
        self.mouseListener.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.kbListener:
            self.kbListener.stop()
        if self.mouseListener:
            self.mouseListener.stop()
