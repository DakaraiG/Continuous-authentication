# capture.py
import sys
import time
import threading
from dataclasses import dataclass
from typing import Optional

from pynput import keyboard, mouse

@dataclass
class CaptureConfig:
    privacyMode: bool = True
    mouseMoveSampleHz: float = 50.0
    enableKeyboard: bool = True

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
        self._kbTap = None
        self._kbTapCallback = None
        self._kbRunLoop = None
        self._kbThread = None

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
    #macOS key compatibility with Quartz
    def _safeMacKeyRepr(self, keycode: int) -> str:
        if self.cfg.privacyMode:
            return "<key>"
        return f"keycode:{keycode}"

    def _startMacKeyboardTap(self):
        try:
            import Quartz  # type: ignore
        except Exception as exc:
            raise RuntimeError("Quartz not available for macOS keyboard capture") from exc

        def _tap_callback(proxy, event_type, event, refcon):
            if event_type in (Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp):
                ts = time.time()
                keycode = int(Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode))
                event_type_str = "down" if event_type == Quartz.kCGEventKeyDown else "up"
                self.db.insertKeyEvent(
                    self.sessionId,
                    ts,
                    event_type_str,
                    self._safeMacKeyRepr(keycode),
                )
            return event

        mask = (1 << Quartz.kCGEventKeyDown) | (1 << Quartz.kCGEventKeyUp)
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            mask,
            _tap_callback,
            None,
        )
        if not tap:
            raise RuntimeError(
                "Keyboard capture unavailable (check macOS Accessibility permissions)"
            )

        self._kbTap = tap
        self._kbTapCallback = _tap_callback

        def _run_loop():
            source = Quartz.CFMachPortCreateRunLoopSource(None, self._kbTap, 0)
            self._kbRunLoop = Quartz.CFRunLoopGetCurrent()
            Quartz.CFRunLoopAddSource(self._kbRunLoop, source, Quartz.kCFRunLoopCommonModes)
            Quartz.CGEventTapEnable(self._kbTap, True)
            Quartz.CFRunLoopRun()

        self._kbThread = threading.Thread(target=_run_loop, daemon=True)
        self._kbThread.start()

    def _stopMacKeyboardTap(self):
        if self._kbRunLoop:
            try:
                import Quartz  # type: ignore
                Quartz.CFRunLoopStop(self._kbRunLoop)
            except Exception:
                pass
        self._kbRunLoop = None
        self._kbTap = None
        self._kbTapCallback = None
        if self._kbThread:
            self._kbThread.join(timeout=0.5)
        self._kbThread = None
    #mouse events
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
        if self.cfg.enableKeyboard:
            if sys.platform == "darwin":
                self._startMacKeyboardTap()
            else:
                self.kbListener = keyboard.Listener(on_press=self.onKeyPress, on_release=self.onKeyRelease)
        self.mouseListener = mouse.Listener(on_move=self.onMove, on_click=self.onClick, on_scroll=self.onScroll)
        if self.kbListener:
            self.kbListener.start()
        self.mouseListener.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        if sys.platform == "darwin":
            self._stopMacKeyboardTap()
        if self.kbListener:
            self.kbListener.stop()
        if self.mouseListener:
            self.mouseListener.stop()
