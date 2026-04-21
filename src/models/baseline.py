import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class BaselineModel:
    def __init__(self, model_type: str = 'logistic_regression'):
        self.model_type = model_type
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        return self.model.predict_proba(X)

    def save(self, filepath: str):
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model.")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, filepath: str):
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
