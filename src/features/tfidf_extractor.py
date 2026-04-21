import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfExtractor:
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True
        )
        self.is_fitted = False

    def fit_transform(self, texts):
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return features

    def transform(self, texts):
        if not self.is_fitted:
            raise ValueError("TfidfExtractor must be fitted before calling transform.")
        return self.vectorizer.transform(texts)

    def save(self, filepath: str):
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted vectorizer.")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load(self, filepath: str):
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Vectorizer file not found at {filepath}")
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
