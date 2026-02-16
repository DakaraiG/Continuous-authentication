# main.py

#crash handler
import faulthandler, sys, traceback
faulthandler.enable(all_threads=True)

def logCrash(excType, exc, tb):
    with open("crash.log", "w", encoding="utf-8") as f:
        f.write("".join(traceback.format_exception(excType, exc, tb)))

sys.excepthook = logCrash



#imports
import sys
import time
import pickle
import threading
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox, QMessageBox,
    QFileDialog, QGroupBox, QProgressBar, QComboBox, QSpinBox,
    QDialog, QDialogButtonBox
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal

from db import Db
from capture import GlobalCapture, CaptureConfig
from features import extractWindowFeatures, featureDictToVector, FEATURE_NAMES
from train_model import (
    buildOneClassDataset, buildBinaryDataset,
    trainOneClass, trainBinary,
    saveModel, loadModel
)


class StepUpDialog(QDialog):
    """
    The step-up prompt that appears when suspicious activity is detected.
    In a real system this would ask for a password or some other verification.
    For the prototype it just asks the user to confirm their identity.
    """
    def __init__(self, consecutiveCount, confidence, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Identity Verification Required")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # warning icon and message
        warningLabel = QLabel(
            "⚠️ Unusual activity detected"
        )
        warningLabel.setStyleSheet("font-size: 18px; font-weight: bold; color: #F44336;")
        warningLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(warningLabel)

        infoLabel = QLabel(
            f"The system has detected {consecutiveCount} consecutive windows of "
            f"activity that don't match the enrolled user's behavioural profile.\n\n"
            f"Latest confidence score: {confidence:.1f}%\n\n"
            f"In a production system, this would trigger a secondary "
            f"authentication challenge (e.g. password or biometric check)."
        )
        infoLabel.setWordWrap(True)
        layout.addWidget(infoLabel)

        # buttons
        buttonBox = QDialogButtonBox()
        self.confirmButton = buttonBox.addButton("I am the enrolled user", QDialogButtonBox.ButtonRole.AcceptRole)
        self.dismissButton = buttonBox.addButton("Dismiss", QDialogButtonBox.ButtonRole.RejectRole)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)


class MainWindow(QMainWindow):
    # signal so the training thread can update the UI safely
    trainingFinished = pyqtSignal(str, object, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Continuous Auth – Capture, Train & Inference")

        self.db = Db()
        self.sessionId = None
        self.capture = None
        self.sessionStartedAt = None

        # model stuff
        self.model = None
        self.modelMetrics = None

        # policy state - tracks how many consecutive low-confidence windows weve seen
        self.consecutiveLowWindows = 0
        self.alertShowing = False  # prevents multiple alerts stacking up

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # ---- capture section ----
        captureGroup = QGroupBox("Data Capture")
        captureLayout = QVBoxLayout(captureGroup)

        self.statusLabel = QLabel("Status: Idle")
        captureLayout.addWidget(self.statusLabel)

        row1 = QHBoxLayout()
        self.userLabelInput = QLineEdit()
        self.userLabelInput.setPlaceholderText("User label (e.g. your name)")
        row1.addWidget(self.userLabelInput)
        captureLayout.addLayout(row1)

        self.privacyModeCheckbox = QCheckBox("Privacy mode (do NOT store key identities)")
        self.privacyModeCheckbox.setChecked(True)
        captureLayout.addWidget(self.privacyModeCheckbox)

        self.keyboardCaptureCheckbox = QCheckBox("Enable keyboard capture (may crash on macOS 15)")
        self.keyboardCaptureCheckbox.setChecked(True)
        captureLayout.addWidget(self.keyboardCaptureCheckbox)

        self.startButton = QPushButton("Start capture (system-wide)")
        self.stopButton = QPushButton("Stop capture")
        self.stopButton.setEnabled(False)

        row2 = QHBoxLayout()
        row2.addWidget(self.startButton)
        row2.addWidget(self.stopButton)
        captureLayout.addLayout(row2)

        self.countsLabel = QLabel("Events: keys=0, mouse=0")
        captureLayout.addWidget(self.countsLabel)

        layout.addWidget(captureGroup)

        # ---- training section ----
        trainGroup = QGroupBox("Model Training")
        trainLayout = QVBoxLayout(trainGroup)

        modelRow = QHBoxLayout()
        modelRow.addWidget(QLabel("Algorithm:"))
        self.modelTypeCombo = QComboBox()
        self.modelTypeCombo.addItem("Isolation Forest (one-class)", "iforest")
        self.modelTypeCombo.addItem("One-Class SVM", "ocsvm")
        self.modelTypeCombo.addItem("Random Forest (needs impostor data)", "rf")
        self.modelTypeCombo.addItem("Logistic Regression (needs impostor data)", "lr")
        modelRow.addWidget(self.modelTypeCombo)
        trainLayout.addLayout(modelRow)

        trainBtnRow = QHBoxLayout()
        self.trainButton = QPushButton("Train Model")
        self.trainButton.setToolTip("Train using all sessions for the user label above")
        trainBtnRow.addWidget(self.trainButton)

        self.saveModelButton = QPushButton("Save Model")
        self.saveModelButton.setEnabled(False)
        trainBtnRow.addWidget(self.saveModelButton)

        self.loadModelButton = QPushButton("Load Model")
        trainBtnRow.addWidget(self.loadModelButton)
        trainLayout.addLayout(trainBtnRow)

        self.trainStatusLabel = QLabel("No model trained yet")
        trainLayout.addWidget(self.trainStatusLabel)

        layout.addWidget(trainGroup)

        # ---- policy section ----
        policyGroup = QGroupBox("Authentication Policy")
        policyLayout = QVBoxLayout(policyGroup)

        # accept threshold
        acceptRow = QHBoxLayout()
        acceptRow.addWidget(QLabel("Accept threshold (%):"))
        self.acceptThresholdSpin = QSpinBox()
        self.acceptThresholdSpin.setRange(1, 99)
        self.acceptThresholdSpin.setValue(60)
        self.acceptThresholdSpin.setToolTip(
            "Confidence above this = user is accepted (green)"
        )
        acceptRow.addWidget(self.acceptThresholdSpin)
        policyLayout.addLayout(acceptRow)

        # challenge threshold
        challengeRow = QHBoxLayout()
        challengeRow.addWidget(QLabel("Challenge threshold (%):"))
        self.challengeThresholdSpin = QSpinBox()
        self.challengeThresholdSpin.setRange(1, 99)
        self.challengeThresholdSpin.setValue(35)
        self.challengeThresholdSpin.setToolTip(
            "Confidence below this = suspicious activity (red)"
        )
        challengeRow.addWidget(self.challengeThresholdSpin)
        policyLayout.addLayout(challengeRow)

        # consecutive windows before alert
        windowsRow = QHBoxLayout()
        windowsRow.addWidget(QLabel("Consecutive low windows before alert:"))
        self.consecutiveWindowsSpin = QSpinBox()
        self.consecutiveWindowsSpin.setRange(1, 20)
        self.consecutiveWindowsSpin.setValue(3)
        self.consecutiveWindowsSpin.setToolTip(
            "How many consecutive suspicious windows before triggering a step-up alert"
        )
        windowsRow.addWidget(self.consecutiveWindowsSpin)
        policyLayout.addLayout(windowsRow)

        self.policyStatusLabel = QLabel("Policy: inactive")
        policyLayout.addWidget(self.policyStatusLabel)

        layout.addWidget(policyGroup)

        # ---- inference section ----
        inferenceGroup = QGroupBox("Live Inference")
        inferenceLayout = QVBoxLayout(inferenceGroup)

        self.confidenceLabel = QLabel("Confidence: —")
        self.confidenceLabel.setStyleSheet("font-size: 16px; font-weight: bold;")
        inferenceLayout.addWidget(self.confidenceLabel)

        self.confidenceBar = QProgressBar()
        self.confidenceBar.setMinimum(0)
        self.confidenceBar.setMaximum(100)
        self.confidenceBar.setValue(0)
        self.confidenceBar.setTextVisible(True)
        self.confidenceBar.setFormat("%v%")
        inferenceLayout.addWidget(self.confidenceBar)

        self.inferenceLogLabel = QLabel("Last inference: —")
        inferenceLayout.addWidget(self.inferenceLogLabel)

        layout.addWidget(inferenceGroup)

        # ---- connect signals ----
        self.startButton.clicked.connect(self.startCapture)
        self.stopButton.clicked.connect(self.stopCapture)
        self.trainButton.clicked.connect(self.trainModel)
        self.saveModelButton.clicked.connect(self.saveModelToFile)
        self.loadModelButton.clicked.connect(self.loadModelFromFile)
        self.trainingFinished.connect(self.onTrainingFinished)

        # timer for refreshing event counts
        self.refreshTimer = QTimer()
        self.refreshTimer.setInterval(500)
        self.refreshTimer.timeout.connect(self.refreshCounts)

        # timer for running inference every 10 seconds
        self.inferenceTimer = QTimer()
        self.inferenceTimer.setInterval(10000)
        self.inferenceTimer.timeout.connect(self.runInference)


    # ---- TRAINING ----

    def trainModel(self):
        """Kick off model training in a background thread so the UI doesnt freeze."""
        userLabel = self.userLabelInput.text().strip()
        if not userLabel:
            QMessageBox.warning(self, "Missing user label",
                "Enter the user label you want to train on (same one you used when capturing).")
            return

        modelTypeData = self.modelTypeCombo.currentData()

        self.trainButton.setEnabled(False)
        self.trainStatusLabel.setText("Training in progress... please wait")

        def doTraining():
            try:
                if modelTypeData in ("iforest", "ocsvm"):
                    X = buildOneClassDataset(userLabel, self.db.path.as_posix())
                    if X is None or len(X) < 5:
                        self.trainingFinished.emit(
                            "Not enough data. Capture at least a few minutes of activity first.",
                            None, None
                        )
                        return
                    model, metrics = trainOneClass(X, modelType=modelTypeData)

                else:
                    X, y = buildBinaryDataset(userLabel, self.db.path.as_posix())
                    if X is None or len(X) < 10:
                        self.trainingFinished.emit(
                            "Not enough data to train a binary model.",
                            None, None
                        )
                        return
                    if len(np.unique(y)) < 2:
                        self.trainingFinished.emit(
                            "Binary mode needs data from both the enrolled user AND at least one impostor. "
                            "Try one-class mode instead, or collect impostor sessions.",
                            None, None
                        )
                        return
                    model, metrics = trainBinary(X, y, modelType=modelTypeData)

                self.trainingFinished.emit("success", model, metrics)

            except Exception as e:
                self.trainingFinished.emit(f"Training failed: {e}", None, None)

        t = threading.Thread(target=doTraining, daemon=True)
        t.start()


    def onTrainingFinished(self, status, model, metrics):
        """Called when training thread finishes (runs on the UI thread)."""
        self.trainButton.setEnabled(True)

        if model is None:
            self.trainStatusLabel.setText(status)
            if "Not enough" in status or "Binary mode" in status:
                QMessageBox.warning(self, "Training issue", status)
            else:
                QMessageBox.critical(self, "Training failed", status)
            return

        self.model = model
        self.modelMetrics = metrics
        self.saveModelButton.setEnabled(True)

        modelType = metrics.get("modelType", "Unknown")
        mode = metrics.get("mode", "unknown")

        if mode == "one-class":
            windows = metrics.get("trainingWindows", 0)
            scoreMean = metrics.get("trainScoreMean", 0)
            inlierRate = metrics.get("inlierRate", 0)
            self.trainStatusLabel.setText(
                f"Trained: {modelType} | {windows} windows | "
                f"avg score: {scoreMean:.2f} | inlier rate: {inlierRate*100:.0f}%"
            )
        else:
            rocAuc = metrics.get("rocAuc", 0)
            eer = metrics.get("eer", 0)
            self.trainStatusLabel.setText(
                f"Trained: {modelType} | AUC: {rocAuc:.3f} | EER: {eer:.3f}"
            )

        QMessageBox.information(self, "Training complete",
            f"Model trained successfully!\n\nType: {modelType}\nMode: {mode}\n\n"
            f"You can now save the model or start capturing with live inference.")


    def saveModelToFile(self):
        if self.model is None:
            return
        filePath, _ = QFileDialog.getSaveFileName(
            self, "Save model", "model.pkl", "Pickle files (*.pkl)"
        )
        if not filePath:
            return
        try:
            saveModel(self.model, self.modelMetrics, filePath)
            QMessageBox.information(self, "Saved", f"Model saved to {filePath}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))


    def loadModelFromFile(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Load model", "", "Pickle files (*.pkl);;All files (*)"
        )
        if not filePath:
            return
        try:
            model, metrics, featureNames = loadModel(filePath)
            self.model = model
            self.modelMetrics = metrics
            self.saveModelButton.setEnabled(True)

            modelType = metrics.get("modelType", "Unknown")
            mode = metrics.get("mode", "unknown")
            self.trainStatusLabel.setText(f"Loaded: {modelType} ({mode} mode)")
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))


    # ---- INFERENCE + POLICY ----

    def runInference(self):
        """
        Run inference on the last 10 seconds of captured data.
        Applies the confidence-gated policy to decide what action to take.
        """
        if self.model is None or self.sessionId is None:
            return

        try:
            now = time.time()
            windowStart = now - 10.0
            windowEnd = now

            self.db.commit()

            # extract features for this window
            featDict = extractWindowFeatures(
                self.db.conn, self.sessionId, windowStart, windowEnd
            )

            totalEvents = featDict.get("totalEvents", 0)
            if totalEvents < 3:
                self.confidenceLabel.setText("Confidence: — (not enough activity)")
                self.inferenceLogLabel.setText(
                    f"Last check: {time.strftime('%H:%M:%S')} - skipped ({totalEvents} events)"
                )
                # dont count idle windows against the user
                return

            # run prediction
            featureVec = featureDictToVector(featDict).reshape(1, -1)
            prob = self.model.predict_proba(featureVec)[0][1]
            confidence = prob * 100

            # get current threshold settings
            acceptThreshold = self.acceptThresholdSpin.value()
            challengeThreshold = self.challengeThresholdSpin.value()
            maxConsecutive = self.consecutiveWindowsSpin.value()

            # apply the confidence policy
            if confidence >= acceptThreshold:
                # user looks legit - reset the counter
                self.consecutiveLowWindows = 0
                policyState = "accepted"
                colour = "#4CAF50"  # green
                self.policyStatusLabel.setText(
                    f"Policy: ✓ User accepted (score {confidence:.0f}% ≥ {acceptThreshold}%)"
                )

            elif confidence >= challengeThreshold:
                # in the warning zone - not quite suspicious enough to alert
                # but we keep an eye on it
                self.consecutiveLowWindows = 0  # reset since its above challenge
                policyState = "warning"
                colour = "#FF9800"  # orange
                self.policyStatusLabel.setText(
                    f"Policy: ⚠ Elevated monitoring (score {confidence:.0f}%)"
                )

            else:
                # below challenge threshold - this looks suspicious
                self.consecutiveLowWindows += 1
                policyState = "suspicious"
                colour = "#F44336"  # red
                self.policyStatusLabel.setText(
                    f"Policy: 🚨 Suspicious ({self.consecutiveLowWindows}/{maxConsecutive} low windows)"
                )

            # log the policy event to the database for later analysis
            self.db.insertPolicyEvent(
                self.sessionId, now, policyState, confidence / 100.0,
                self.consecutiveLowWindows, acceptThreshold / 100.0,
                challengeThreshold / 100.0,
            )

            # update confidence display
            self.confidenceBar.setValue(int(confidence))
            self.confidenceLabel.setText(f"Confidence: {confidence:.1f}%")
            self.confidenceBar.setStyleSheet(
                f"QProgressBar::chunk {{ background-color: {colour}; }}"
            )

            self.inferenceLogLabel.setText(
                f"Last check: {time.strftime('%H:%M:%S')} - "
                f"score={confidence:.1f}%, events={totalEvents}, state={policyState}"
            )

            # trigger step-up alert if weve hit the threshold
            if self.consecutiveLowWindows >= maxConsecutive and not self.alertShowing:
                self.triggerStepUpAlert(confidence)

        except Exception as e:
            self.inferenceLogLabel.setText(f"Inference error: {e}")
            print(f"Inference error: {e}")


    def triggerStepUpAlert(self, confidence):
        """
        Show the step-up authentication prompt.
        In a real system this would require the user to re-authenticate.
        For the prototype we just show a dialog and log the event.
        """
        self.alertShowing = True

        # log the alert event
        self.db.insertPolicyEvent(
            self.sessionId, time.time(), "alert", confidence / 100.0,
            self.consecutiveLowWindows,
            self.acceptThresholdSpin.value() / 100.0,
            self.challengeThresholdSpin.value() / 100.0,
        )
        self.db.commit()

        self.policyStatusLabel.setText("Policy: 🔒 STEP-UP AUTHENTICATION REQUIRED")

        dialog = StepUpDialog(self.consecutiveLowWindows, confidence, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            # user confirmed their identity - reset the counter
            self.consecutiveLowWindows = 0
            self.policyStatusLabel.setText("Policy: ✓ Identity confirmed by user")

            # log that the user confirmed
            self.db.insertPolicyEvent(
                self.sessionId, time.time(), "confirmed", confidence / 100.0,
                0, self.acceptThresholdSpin.value() / 100.0,
                self.challengeThresholdSpin.value() / 100.0,
            )
        else:
            # user dismissed - keep monitoring but reset counter to avoid spamming
            self.consecutiveLowWindows = 0
            self.policyStatusLabel.setText("Policy: ⚠ Alert dismissed, monitoring continues")

            self.db.insertPolicyEvent(
                self.sessionId, time.time(), "dismissed", confidence / 100.0,
                0, self.acceptThresholdSpin.value() / 100.0,
                self.challengeThresholdSpin.value() / 100.0,
            )

        self.db.commit()
        self.alertShowing = False


    # ---- CAPTURE ----

    def startCapture(self):
        try:
            userLabel = self.userLabelInput.text().strip()
            if not userLabel:
                QMessageBox.warning(self, "Missing user label",
                    "Please enter a user label before starting.")
                return

            self.sessionStartedAt = time.time()
            self.sessionId = self.db.startSession(
                userLabel=userLabel, mode="global", startedAt=self.sessionStartedAt
            )

            cfg = CaptureConfig(
                privacyMode=self.privacyModeCheckbox.isChecked(),
                mouseMoveSampleHz=20.0,
                enableKeyboard=self.keyboardCaptureCheckbox.isChecked(),
            )
            self.capture = GlobalCapture(self.db, self.sessionId, cfg)
            self.capture.start()

            # reset policy state for new session
            self.consecutiveLowWindows = 0
            self.alertShowing = False

            self.statusLabel.setText(f"Status: Capturing (session {self.sessionId})")
            self.startButton.setEnabled(False)
            self.stopButton.setEnabled(True)
            self.userLabelInput.setEnabled(False)
            self.refreshTimer.start()

            # start live inference if we have a model
            if self.model is not None:
                self.inferenceTimer.start()
                self.inferenceLogLabel.setText("Inference active — waiting for first window...")
                self.policyStatusLabel.setText("Policy: active, monitoring...")

        except Exception as e:
            try:
                if self.capture:
                    self.capture.stop()
            except Exception:
                pass
            try:
                if self.sessionId is not None:
                    self.db.endSession(self.sessionId, time.time())
            except Exception:
                pass

            QMessageBox.critical(self, "Capture failed", f"{type(e).__name__}: {e}")
            self.capture = None
            self.sessionId = None
            self.sessionStartedAt = None


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
        self.userLabelInput.setEnabled(True)
        self.refreshTimer.stop()
        self.inferenceTimer.stop()
        self.refreshCounts()

        self.policyStatusLabel.setText("Policy: inactive")
        self.consecutiveLowWindows = 0

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


#main
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(640, 580)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()