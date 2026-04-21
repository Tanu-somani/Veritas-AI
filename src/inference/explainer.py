import shap
import numpy as np

class ModelExplainer:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        # We need the underlying sklearn model
        self.sklearn_model = model.model
        
        # Use LinearExplainer for Logistic Regression or TreeExplainer for Random Forest
        if hasattr(self.sklearn_model, "coef_"):
            # It's a linear model
            self.explainer = shap.LinearExplainer(
                self.sklearn_model, 
                shap.maskers.Independent(np.zeros((1, len(vectorizer.vectorizer.vocabulary_))))
            )
        else:
            # Tree based model fallback
            self.explainer = shap.TreeExplainer(self.sklearn_model)
            
        self.feature_names = self.vectorizer.vectorizer.get_feature_names_out()

    def explain_prediction(self, text: str, cleaned_text: str):
        """
        Returns the top contributing words for the prediction.
        """
        features = self.vectorizer.transform([cleaned_text])
        
        # LinearExplainer might need dense array or we can just extract coefficients
        # For simplicity in a web app, let's just use the coefficients directly if it's logistic regression
        # This is a robust manual explanation logic that works reliably for Linear Models:
        if hasattr(self.sklearn_model, "coef_"):
            feature_array = features.toarray()[0]
            coefs = self.sklearn_model.coef_[0]
            
            contributions = feature_array * coefs
            
            # Get indices of non-zero features
            non_zero_idx = np.nonzero(feature_array)[0]
            
            word_impacts = []
            for idx in non_zero_idx:
                word_impacts.append((self.feature_names[idx], contributions[idx]))
                
            # Sort by absolute impact
            word_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Format explanation
            explanation = "Top contributing words: "
            top_words = []
            for word, impact in word_impacts[:5]:
                direction = "FAKE" if impact < 0 else "REAL"
                top_words.append(f"'{word}' (pushed towards {direction})")
            
            explanation += ", ".join(top_words)
            return explanation
        else:
            return "Explainability for this model type is not fully configured."
