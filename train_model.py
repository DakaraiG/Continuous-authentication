# train_model.py
# Trains authentication models for continuous auth
# Supports one-class (enrolled user only) and binary (enrolled + impostor) modes
#
# v2 - better hyperparameters, training data filtering, improved score calibration

import pickle
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from features import extractAllFeatures, featureDictToVector, FEATURE_NAMES


def buildOneClassDataset(enrolledUser, dbPath="auth_log.db"):
    """
    Build dataset for one-class training (only enrolled user data needed).
    Filters out low-quality windows automatically via the minEvents param in features.
    """
    print(f"Extracting features for enrolled user: {enrolledUser}")
    allData = extractAllFeatures(dbPath, minEvents=10)

    X = []
    for userLabel, featDict in allData:
        if userLabel.lower() == enrolledUser.lower():
            vec = featureDictToVector(featDict)
            X.append(vec)

    if len(X) == 0:
        print(f"No data found for user '{enrolledUser}'!")
        return None

    X = np.array(X)

    # remove any rows that are all zeros (completely empty windows that slipped through)
    rowSums = np.sum(np.abs(X), axis=1)
    validMask = rowSums > 0
    X = X[validMask]

    print(f"Dataset: {len(X)} usable enrolled windows")
    return X


def buildBinaryDataset(enrolledUser, dbPath="auth_log.db"):
    """
    Build dataset for binary classification (needs enrolled + impostor data).
    """
    print(f"Extracting features for enrolled user: {enrolledUser}")
    allData = extractAllFeatures(dbPath, minEvents=10)

    if len(allData) == 0:
        return None, None

    X, y = [], []
    enrolledCount, impostorCount = 0, 0

    for userLabel, featDict in allData:
        vec = featureDictToVector(featDict)
        X.append(vec)
        if userLabel.lower() == enrolledUser.lower():
            y.append(1)
            enrolledCount += 1
        else:
            y.append(0)
            impostorCount += 1

    X = np.array(X)
    y = np.array(y)
    print(f"Dataset: {enrolledCount} enrolled, {impostorCount} impostor windows")
    return X, y


class OneClassModel:
    """
    Wraps a one-class classifier with proper score calibration.
    Trains on enrolled user data only and flags anomalies.
    
    The key improvement here is better score normalisation - we use
    the full training distribution rather than just percentiles so
    the confidence scores are more meaningful.
    """
    def __init__(self, modelType="iforest"):
        self.modelType = modelType
        self.scaler = StandardScaler()
        self.model = None
        self.scoreMean = None
        self.scoreStd = None

    def fit(self, X):
        """Train on enrolled user data only."""
        xScaled = self.scaler.fit_transform(X)

        if self.modelType == "ocsvm":
            # nu controls the upper bound on training errors and lower bound on support vectors
            # lower nu = tighter boundary around normal data
            self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        else:
            # isolation forest
            # lower contamination = assume less noise in training data
            # more estimators = more stable predictions
            self.model = IsolationForest(
                n_estimators=200,
                contamination=0.03,
                max_features=0.8,  # use subset of features per tree for diversity
                random_state=42,
            )

        self.model.fit(xScaled)

        # calibrate scores using the training distribution
        # using mean/std gives a more stable normalisation than min/max or percentiles
        scores = self.model.decision_function(xScaled)
        self.scoreMean = np.mean(scores)
        self.scoreStd = np.std(scores)

        return self

    def predict_proba(self, X):
        """
        Return probability-like scores mapped to 0-1 range.
        Uses z-score normalisation based on training distribution,
        then sigmoid to squeeze into 0-1. This gives smoother and more
        stable confidence values compared to min-max scaling.
        """
        xScaled = self.scaler.transform(X)
        rawScores = self.model.decision_function(xScaled)

        # z-score normalise using training stats
        zScores = (rawScores - self.scoreMean) / (self.scoreStd + 1e-8)

        # sigmoid function to map to 0-1
        # the * 1.5 stretches the sigmoid so small deviations dont all collapse to 0.5
        normed = 1.0 / (1.0 + np.exp(-zScores * 1.5))
        normed = np.clip(normed, 0.0, 1.0)

        # format like sklearn: col0=impostor, col1=enrolled
        probs = np.column_stack([1 - normed, normed])
        return probs

    def predict(self, X):
        """Predict 1 (enrolled) or 0 (anomaly)."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


def trainOneClass(X, modelType="iforest"):
    """
    Train a one-class model. Only needs enrolled user data.
    """
    if modelType == "ocsvm":
        modelName = "One-Class SVM"
    else:
        modelName = "Isolation Forest"

    print(f"\nTraining {modelName} (one-class mode)...")
    print(f"  Features: {len(FEATURE_NAMES)}")
    print(f"  Training windows: {len(X)}")

    model = OneClassModel(modelType=modelType)
    model.fit(X)

    # score distribution on training data
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    inliers = np.sum(preds == 1)

    print(f"  Training score distribution:")
    print(f"    mean: {np.mean(probs):.3f}")
    print(f"    std:  {np.std(probs):.3f}")
    print(f"    min:  {np.min(probs):.3f}")
    print(f"    max:  {np.max(probs):.3f}")
    print(f"  Inliers: {inliers}/{len(X)} ({100*inliers/len(X):.1f}%)")

    metrics = {
        "modelType": modelName,
        "mode": "one-class",
        "featureCount": len(FEATURE_NAMES),
        "trainingWindows": len(X),
        "trainScoreMean": float(np.mean(probs)),
        "trainScoreStd": float(np.std(probs)),
        "trainScoreMin": float(np.min(probs)),
        "trainScoreMax": float(np.max(probs)),
        "inlierRate": float(inliers / len(X)),
    }

    return model, metrics


def trainBinary(X, y, modelType="rf"):
    """
    Train a binary classifier with cross-validation.
    Needs both enrolled and impostor data.
    """
    if modelType == "lr":
        baseModel = LogisticRegression(max_iter=1000, random_state=42)
        modelName = "Logistic Regression"
    elif modelType == "svm":
        baseModel = SVC(kernel="rbf", probability=False, random_state=42)
        modelName = "SVM (RBF)"
    else:
        baseModel = RandomForestClassifier(n_estimators=100, random_state=42)
        modelName = "Random Forest"

    print(f"\nTraining {modelName} (binary mode)...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", CalibratedClassifierCV(baseModel, cv=3, method="sigmoid")),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    allProbs = np.zeros(len(y))
    allPreds = np.zeros(len(y))

    for foldIdx, (trainIdx, testIdx) in enumerate(skf.split(X, y)):
        xTrain, xTest = X[trainIdx], X[testIdx]
        yTrain, yTest = y[trainIdx], y[testIdx]
        pipeline.fit(xTrain, yTrain)
        allProbs[testIdx] = pipeline.predict_proba(xTest)[:, 1]
        allPreds[testIdx] = pipeline.predict(xTest)

    fpr, tpr, thresholds = roc_curve(y, allProbs)
    rocAuc = auc(fpr, tpr)

    fnr = 1 - tpr
    eerIdx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eerIdx] + fnr[eerIdx]) / 2
    eerThreshold = thresholds[eerIdx]

    cm = confusion_matrix(y, allPreds)

    print(f"  ROC-AUC: {rocAuc:.4f}")
    print(f"  EER: {eer:.4f}")

    pipeline.fit(X, y)

    metrics = {
        "modelType": modelName,
        "mode": "binary",
        "rocAuc": rocAuc,
        "eer": eer,
        "eerThreshold": eerThreshold,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "confusionMatrix": cm.tolist(),
    }

    return pipeline, metrics


def saveModel(model, metrics, outputPath="model.pkl"):
    """Save the trained model and metrics."""
    data = {
        "pipeline": model,
        "metrics": metrics,
        "featureNames": FEATURE_NAMES,
    }
    with open(outputPath, "wb") as f:
        pickle.dump(data, f)
    print(f"Model saved to {outputPath}")


def loadModel(modelPath="model.pkl"):
    """Load a saved model."""
    with open(modelPath, "rb") as f:
        data = pickle.load(f)
    return data["pipeline"], data["metrics"], data["featureNames"]