# main.py
import sys
import time

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox, QMessageBox
)
from PyQt6.QtCore import QTimer

from db import Db
from capture import GlobalCapture, CaptureConfig

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Continuous Auth – Capture MVP")

        self.db = Db()
        self.sessionId = None
        self.capture = None
        self.sessionStartedAt = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.statusLabel = QLabel("Status: Idle")
        layout.addWidget(self.statusLabel)

        row1 = QHBoxLayout()
        self.userLabelInput = QLineEdit()
        self.userLabelInput.setPlaceholderText("User label (e.g., Dakarai / Participant1)")
        row1.addWidget(self.userLabelInput)
        layout.addLayout(row1)

        self.privacyModeCheckbox = QCheckBox("Privacy mode (do NOT store key identities)")
        self.privacyModeCheckbox.setChecked(True)
        layout.addWidget(self.privacyModeCheckbox)

        self.startButton = QPushButton("Start capture (system-wide)")
        self.stopButton = QPushButton("Stop capture")
        self.stopButton.setEnabled(False)

        row2 = QHBoxLayout()
        row2.addWidget(self.startButton)
        row2.addWidget(self.stopButton)
        layout.addLayout(row2)

        self.countsLabel = QLabel("Events: keys=0, mouse=0")
        layout.addWidget(self.countsLabel)

        self.startButton.clicked.connect(self.startCapture)
        self.stopButton.clicked.connect(self.stopCapture)

        self.refreshTimer = QTimer()
        self.refreshTimer.setInterval(500)
        self.refreshTimer.timeout.connect(self.refreshCounts)

    def startCapture(self):
        userLabel = self.userLabelInput.text().strip()
        if not userLabel:
            QMessageBox.warning(self, "Missing user label", "Please enter a user label before starting.")
            return

        self.sessionStartedAt = time.time()
        self.sessionId = self.db.startSession(
            userLabel=userLabel,
            mode="global",
            startedAt=self.sessionStartedAt
        )

        cfg = CaptureConfig(
            privacyMode=self.privacyModeCheckbox.isChecked(),
            mouseMoveSampleHz=50.0
        )
        self.capture = GlobalCapture(self.db, self.sessionId, cfg)
        self.capture.start()

        self.statusLabel.setText(f"Status: Capturing (session {self.sessionId})")
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.refreshTimer.start()

    def stopCapture(self):
        if self.capture:
            self.capture.stop()

        endedAt = time.time()
        if self.sessionId is not None:
            self.db.commit()
            self.db.endSession(self.sessionId, endedAt)

        self.statusLabel.setText("Status: Idle (saved to auth_log.db)")
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.refreshTimer.stop()
        self.refreshCounts()

        self.capture = None
        self.sessionId = None
        self.sessionStartedAt = None

    def refreshCounts(self):
        if self.sessionId is None:
            return
        self.db.commit()
        keyCount, mouseCount = self.db.getEventCountsForSession(self.sessionId)
        self.countsLabel.setText(f"Events: keys={keyCount}, mouse={mouseCount}")

    def closeEvent(self, event):
        try:
            if self.capture:
                self.capture.stop()
            self.db.commit()
            self.db.close()
        except Exception:
            pass
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(640, 220)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
