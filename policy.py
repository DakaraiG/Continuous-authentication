# policy.py
# Handles the confidence-gated authentication policy
# - threshold checks (accept / warning / suspicious)
# - consecutive window tracking
# - step-up alert dialog
# - model adaptation after user confirmation

import time
import threading
import numpy as np

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox, QMessageBox
from PyQt6.QtCore import Qt

from train_model import (
    buildOneClassDataset, buildBinaryDataset,
    trainOneClass, trainBinary,
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

        warningLabel = QLabel("Unusual activity detected")
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

        buttonBox = QDialogButtonBox()
        self.confirmButton = buttonBox.addButton(
            "I am the enrolled user", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.dismissButton = buttonBox.addButton(
            "Dismiss", QDialogButtonBox.ButtonRole.RejectRole
        )
        # connect buttons directly to avoid PyQt6 signal quirk with
        # AcceptRole/RejectRole that can require multiple clicks
        self.confirmButton.clicked.connect(self.accept)
        self.dismissButton.clicked.connect(self.reject)
        layout.addWidget(buttonBox)


class PolicyEngine:
    """
    Manages the confidence-gated authentication policy.
    
    Keeps track of consecutive low-confidence windows and decides
    when to trigger step-up alerts. Also handles model adaptation
    after the user confirms their identity.
    """
    def __init__(self):
        self.consecutiveLowWindows = 0
        self.alertShowing = False
        self.confirmationCount = 0
        self.confirmationsBeforeRetrain = 1

    def reset(self):
        """Reset policy state (e.g. when starting a new session)."""
        self.consecutiveLowWindows = 0
        self.alertShowing = False
        self.confirmationCount = 0

    def evaluate(self, confidence, acceptThreshold, challengeThreshold):
        """
        Evaluate a confidence score against the policy thresholds.
        
        Returns one of: "accepted", "warning", "suspicious"
        """
        if confidence >= acceptThreshold:
            self.consecutiveLowWindows = 0
            return "accepted"
        elif confidence >= challengeThreshold:
            self.consecutiveLowWindows = 0
            return "warning"
        else:
            self.consecutiveLowWindows += 1
            return "suspicious"

    def shouldAlert(self, maxConsecutive):
        """Check if we should trigger a step-up alert."""
        return self.consecutiveLowWindows >= maxConsecutive and not self.alertShowing

    def showStepUpAlert(self, confidence, sessionId, db,
                        acceptThreshold, challengeThreshold, parent=None):
        """
        Show the step-up dialog and handle the users response.
        Returns True if the user confirmed their identity, False otherwise.
        """
        self.alertShowing = True

        # log the alert
        db.insertPolicyEvent(
            sessionId, time.time(), "alert", confidence / 100.0,
            self.consecutiveLowWindows, acceptThreshold, challengeThreshold,
        )
        db.commit()

        dialog = StepUpDialog(self.consecutiveLowWindows, confidence, parent=parent)
        result = dialog.exec()

        confirmed = (result == QDialog.DialogCode.Accepted)

        if confirmed:
            self.consecutiveLowWindows = 0
            self.confirmationCount += 1

            db.insertPolicyEvent(
                sessionId, time.time(), "confirmed", confidence / 100.0,
                0, acceptThreshold, challengeThreshold,
            )
        else:
            # dismissed - reset counter to avoid spamming alerts
            self.consecutiveLowWindows = 0

            db.insertPolicyEvent(
                sessionId, time.time(), "dismissed", confidence / 100.0,
                0, acceptThreshold, challengeThreshold,
            )

        db.commit()
        self.alertShowing = False
        return confirmed

    def shouldAdapt(self):
        """Check if enough confirmations have happened to trigger a retrain."""
        return self.confirmationCount >= self.confirmationsBeforeRetrain

    def triggerAdaptation(self, userLabel, dbPath, currentModel, currentMetrics,
                          trainingFinishedSignal):
        """
        Retrain the model in a background thread using all available data
        including the current session. Called when the user has confirmed
        their identity enough times.
        """
        self.confirmationCount = 0  # reset so we dont retrain every confirmation

        # figure out what model type to use based on the current model
        modelType = "iforest"
        if currentMetrics:
            mType = currentMetrics.get("modelType", "")
            if "SVM" in mType:
                modelType = "ocsvm"
            elif "Random Forest" in mType and currentMetrics.get("mode") == "binary":
                modelType = "rf"
            elif "Logistic" in mType:
                modelType = "lr"

        def doAdapt():
            try:
                if modelType in ("iforest", "ocsvm"):
                    X = buildOneClassDataset(userLabel, dbPath)
                    if X is None or len(X) < 5:
                        trainingFinishedSignal.emit(
                            "Adaptation failed: not enough data", None, None
                        )
                        return
                    model, metrics = trainOneClass(X, modelType=modelType)
                else:
                    X, y = buildBinaryDataset(userLabel, dbPath)
                    if X is None or len(np.unique(y)) < 2:
                        trainingFinishedSignal.emit(
                            "Adaptation failed: insufficient data", None, None
                        )
                        return
                    model, metrics = trainBinary(X, y, modelType=modelType)

                metrics["adapted"] = True
                trainingFinishedSignal.emit("success", model, metrics)

            except Exception as e:
                trainingFinishedSignal.emit(f"Adaptation failed: {e}", None, None)

        t = threading.Thread(target=doAdapt, daemon=True)
        t.start()