import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class RFModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X, y):
        print("Training Random Forest...")
        self.model.fit(X, y)
        print("Random Forest training complete.")

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X, y, scenario=''):
        proba = self.predict_proba(X)
        preds = (proba >= 0.5).astype(int)
        metrics = {
            'Accuracy': accuracy_score(y, preds),
            'Precision': precision_score(y, preds, zero_division=0),
            'Recall': recall_score(y, preds, zero_division=0),
            'F1': f1_score(y, preds, zero_division=0),
            'ROC_AUC': roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.0,
        }
        if scenario:
            print(f"[RF {scenario}] {metrics}")
        return metrics, proba

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
