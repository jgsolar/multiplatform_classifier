import time
import numpy as np
from sklearn.metrics import confusion_matrix


def train_and_time(model_search, X, y):
    start_time = time.time()
    model_search.fit(X, y)
    end_time = time.time()
    duration = end_time - start_time
    return duration


def custom_specificity_multiclass(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    specificity = TN / (TN + FP)
    return np.mean(specificity)
