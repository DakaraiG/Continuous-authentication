# window.py
# Main application window - handles capture, training UI, and live inference display
# Policy logic is handled by PolicyEngine in policy.py

import time
import pickle
import threading
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox, QMessageBox,
    QFileDialog, QGroupBox, QProgressBar, QComboBox, QSpinBox
)
from PyQt6.QtCore import QTimer, pyqtSignal

from db import Db
from capture import GlobalCapture, CaptureConfig
from features import extractWindowFeatures, featureDictToVector
from train_model import (
    buildOneClassDataset, buildBinaryDataset,
    trainOneClass, trainBinary,
    saveModel, loadModel
)
from policy import PolicyEngine


class MainWindow(QMainWindow):
    trainingFinished = pyqtSignal(str, object, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Continuous Auth – Capture, Train & Inference")

        self.db = Db()
        self.sessionId = None
        self.capture = None
        self.sessionStartedAt = None

        # model
        self.model = None
        self.modelMetrics = None

        # policy engine handles all threshold logic and alerts
        self.policy = PolicyEngine()

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.buildCaptureSection(layout)
        self.buildTrainingSection(layout)
        self.buildPolicySection(layout)
        self.buildInferenceSection(layout)

        # connect signals
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


    # ---- UI BUILDING ----

    def buildCaptureSection(self, parentLayout):
        group = QGroupBox("Data Capture")
        layout = QVBoxLayout(group)

        self.statusLabel = QLabel("Status: Idle")
        layout.addWidget(self.statusLabel)

        row1 = QHBoxLayout()
        self.userLabelInput = QLineEdit()
        self.userLabelInput.setPlaceholderText("User label (e.g. your name)")
        row1.addWidget(self.userLabelInput)
        layout.addLayout(row1)

        self.privacyModeCheckbox = QCheckBox("Privacy mode (do NOT store key identities)")
        self.privacyModeCheckbox.setChecked(True)
        layout.addWidget(self.privacyModeCheckbox)

        self.keyboardCaptureCheckbox = QCheckBox("Enable keyboard capture (may crash on macOS 15)")
        self.keyboardCaptureCheckbox.setChecked(True)
        layout.addWidget(self.keyboardCaptureCheckbox)

        self.startButton = QPushButton("Start capture (system-wide)")
        self.stopButton = QPushButton("Stop capture")
        self.stopButton.setEnabled(False)

        row2 = QHBoxLayout()
        row2.addWidget(self.startButton)
        row2.addWidget(self.stopButton)
        layout.addLayout(row2)

        self.countsLabel = QLabel("Events: keys=0, mouse=0")
        layout.addWidget(self.countsLabel)

        parentLayout.addWidget(group)


    def buildTrainingSection(self, parentLayout):
        group = QGroupBox("Model Training")
        layout = QVBoxLayout(group)

        modelRow = QHBoxLayout()
        modelRow.addWidget(QLabel("Algorithm:"))
        self.modelTypeCombo = QComboBox()
        self.modelTypeCombo.addItem("Isolation Forest (Large dataset)", "iforest")
        self.modelTypeCombo.addItem("One-Class SVM (Small dataset)", "ocsvm")
        modelRow.addWidget(self.modelTypeCombo)
        layout.addLayout(modelRow)

        btnRow = QHBoxLayout()
        self.trainButton = QPushButton("Train Model")
        self.trainButton.setToolTip("Train using all sessions for the user label above")
        btnRow.addWidget(self.trainButton)

        self.saveModelButton = QPushButton("Save Model")
        self.saveModelButton.setEnabled(False)
        btnRow.addWidget(self.saveModelButton)

        self.loadModelButton = QPushButton("Load Model")
        btnRow.addWidget(self.loadModelButton)
        layout.addLayout(btnRow)

        self.trainStatusLabel = QLabel("No model trained yet")
        layout.addWidget(self.trainStatusLabel)

        parentLayout.addWidget(group)


    def buildPolicySection(self, parentLayout):
        group = QGroupBox("Authentication Policy")
        layout = QVBoxLayout(group)

        acceptRow = QHBoxLayout()
        acceptRow.addWidget(QLabel("Accept threshold (%):"))
        self.acceptThresholdSpin = QSpinBox()
        self.acceptThresholdSpin.setRange(1, 99)
        self.acceptThresholdSpin.setValue(60)
        self.acceptThresholdSpin.setToolTip("Confidence above this = user accepted (green)")
        acceptRow.addWidget(self.acceptThresholdSpin)
        layout.addLayout(acceptRow)

        challengeRow = QHBoxLayout()
        challengeRow.addWidget(QLabel("Challenge threshold (%):"))
        self.challengeThresholdSpin = QSpinBox()
        self.challengeThresholdSpin.setRange(1, 99)
        self.challengeThresholdSpin.setValue(35)
        self.challengeThresholdSpin.setToolTip("Confidence below this = suspicious (red)")
        challengeRow.addWidget(self.challengeThresholdSpin)
        layout.addLayout(challengeRow)

        windowsRow = QHBoxLayout()
        windowsRow.addWidget(QLabel("Consecutive low windows before alert:"))
        self.consecutiveWindowsSpin = QSpinBox()
        self.consecutiveWindowsSpin.setRange(1, 20)
        self.consecutiveWindowsSpin.setValue(3)
        self.consecutiveWindowsSpin.setToolTip("How many suspicious windows before triggering a step-up alert")
        windowsRow.addWidget(self.consecutiveWindowsSpin)
        layout.addLayout(windowsRow)

        self.policyStatusLabel = QLabel("Policy: inactive")
        layout.addWidget(self.policyStatusLabel)

        parentLayout.addWidget(group)


    def buildInferenceSection(self, parentLayout):
        group = QGroupBox("Live Inference")
        layout = QVBoxLayout(group)

        self.confidenceLabel = QLabel("Confidence: —")
        self.confidenceLabel.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.confidenceLabel)

        self.confidenceBar = QProgressBar()
        self.confidenceBar.setMinimum(0)
        self.confidenceBar.setMaximum(100)
        self.confidenceBar.setValue(0)
        self.confidenceBar.setTextVisible(True)
        self.confidenceBar.setFormat("%v%")
        layout.addWidget(self.confidenceBar)

        self.inferenceLogLabel = QLabel("Last inference: —")
        layout.addWidget(self.inferenceLogLabel)

        parentLayout.addWidget(group)


    # ---- TRAINING ----

    def trainModel(self):
        """Kick off model training in a background thread."""
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
                            None, None)
                        return
                    model, metrics = trainOneClass(X, modelType=modelTypeData)
                else:
                    X, y = buildBinaryDataset(userLabel, self.db.path.as_posix())
                    if X is None or len(X) < 10:
                        self.trainingFinished.emit(
                            "Not enough data to train a binary model.", None, None)
                        return
                    if len(np.unique(y)) < 2:
                        self.trainingFinished.emit(
                            "Binary mode needs both enrolled and impostor data. "
                            "Try one-class mode instead.", None, None)
                        return
                    model, metrics = trainBinary(X, y, modelType=modelTypeData)

                self.trainingFinished.emit("success", model, metrics)
            except Exception as e:
                self.trainingFinished.emit(f"Training failed: {e}", None, None)

        t = threading.Thread(target=doTraining, daemon=True)
        t.start()


    def onTrainingFinished(self, status, model, metrics):
        """Called when training thread finishes (on the UI thread)."""
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

        # update the training status label
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

        # if this was an adaptation, prompt to save
        if metrics.get("adapted", False):
            self.policyStatusLabel.setText(
                "Policy: ✓ Model adapted to include confirmed behaviour"
            )
            reply = QMessageBox.question(
                self, "Model Adapted",
                "The model has been retrained to include your recently confirmed behaviour.\n\n"
                "Would you like to save the updated model now?\n"
                "(If you don't save, the adaptation will be lost when you close the app)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.saveModelToFile()
        else:
            QMessageBox.information(self, "Training complete",
                f"Model trained successfully!\n\nType: {modelType}\nMode: {mode}\n\n"
                f"You can now save the model or start capturing with live inference.")


    def saveModelToFile(self):
        if self.model is None:
            return
        filePath, _ = QFileDialog.getSaveFileName(
            self, "Save model", "model.pkl", "Pickle files (*.pkl)")
        if not filePath:
            return
        try:
            saveModel(self.model, self.modelMetrics, filePath)
            QMessageBox.information(self, "Saved", f"Model saved to {filePath}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))


    def loadModelFromFile(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Load model", "", "Pickle files (*.pkl);;All files (*)")
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
        """Run inference on the last 10s of captured data and apply the policy."""
        if self.model is None or self.sessionId is None:
            return

        try:
            now = time.time()
            windowStart = now - 10.0
            windowEnd = now

            self.db.commit()

            featDict = extractWindowFeatures(
                self.db.conn, self.sessionId, windowStart, windowEnd
            )

            totalEvents = featDict.get("totalEvents", 0)
            if totalEvents < 3:
                self.confidenceLabel.setText("Confidence: — (not enough activity)")
                self.inferenceLogLabel.setText(
                    f"Last check: {time.strftime('%H:%M:%S')} - skipped ({totalEvents} events)")
                return

            # run prediction
            featureVec = featureDictToVector(featDict).reshape(1, -1)
            prob = self.model.predict_proba(featureVec)[0][1]
            confidence = prob * 100

            # get thresholds from the UI
            acceptThreshold = self.acceptThresholdSpin.value()
            challengeThreshold = self.challengeThresholdSpin.value()
            maxConsecutive = self.consecutiveWindowsSpin.value()

            # evaluate against the policy
            policyState = self.policy.evaluate(confidence, acceptThreshold, challengeThreshold)

            # log the event
            self.db.insertPolicyEvent(
                self.sessionId, now, policyState, confidence / 100.0,
                self.policy.consecutiveLowWindows,
                acceptThreshold / 100.0, challengeThreshold / 100.0,
            )

            # update the UI
            self.updateConfidenceDisplay(confidence, policyState, totalEvents,
                                         acceptThreshold, maxConsecutive)

            # check if we need to trigger a step-up alert
            if self.policy.shouldAlert(maxConsecutive):
                confirmed = self.policy.showStepUpAlert(
                    confidence, self.sessionId, self.db,
                    acceptThreshold / 100.0, challengeThreshold / 100.0,
                    parent=self,
                )

                if confirmed:
                    self.policyStatusLabel.setText("Policy: ✓ Identity confirmed by user")

                    # trigger adaptation if needed
                    if self.policy.shouldAdapt():
                        self.policyStatusLabel.setText(
                            "Policy:Adapting model to confirmed behaviour...")
                        self.db.commit()

                        userLabel = self.userLabelInput.text().strip()
                        self.policy.triggerAdaptation(
                            userLabel, self.db.path.as_posix(),
                            self.model, self.modelMetrics,
                            self.trainingFinished,
                        )
                else:
                    self.policyStatusLabel.setText("Policy: ⚠ Alert dismissed, monitoring continues")

        except Exception as e:
            self.inferenceLogLabel.setText(f"Inference error: {e}")
            print(f"Inference error: {e}")


    def updateConfidenceDisplay(self, confidence, policyState, totalEvents,
                                 acceptThreshold, maxConsecutive):
        """Update all the inference-related UI elements."""
        self.confidenceBar.setValue(int(confidence))
        self.confidenceLabel.setText(f"Confidence: {confidence:.1f}%")

        if policyState == "accepted":
            colour = "#4CAF50"
            self.policyStatusLabel.setText(
                f"Policy: ✓ User accepted (score {confidence:.0f}% ≥ {acceptThreshold}%)")
        elif policyState == "warning":
            colour = "#FF9800"
            self.policyStatusLabel.setText(
                f"Policy: ⚠ Elevated monitoring (score {confidence:.0f}%)")
        else:
            colour = "#F44336"
            self.policyStatusLabel.setText(
                f"Policy: 🚨 Suspicious ({self.policy.consecutiveLowWindows}/{maxConsecutive} low windows)")

        self.confidenceBar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {colour}; }}")

        self.inferenceLogLabel.setText(
            f"Last check: {time.strftime('%H:%M:%S')} - "
            f"score={confidence:.1f}%, events={totalEvents}, state={policyState}")


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
                userLabel=userLabel, mode="global", startedAt=self.sessionStartedAt)

            cfg = CaptureConfig(
                privacyMode=self.privacyModeCheckbox.isChecked(),
                mouseMoveSampleHz=20.0,
                enableKeyboard=self.keyboardCaptureCheckbox.isChecked(),
            )
            self.capture = GlobalCapture(self.db, self.sessionId, cfg)
            self.capture.start()

            self.policy.reset()

            self.statusLabel.setText(f"Status: Capturing (session {self.sessionId})")
            self.startButton.setEnabled(False)
            self.stopButton.setEnabled(True)
            self.userLabelInput.setEnabled(False)
            self.refreshTimer.start()

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
        self.policy.reset()

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