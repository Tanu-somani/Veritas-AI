import os
import pandas as pd
from pathlib import Path
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.text_cleaner import TextCleaner
from src.features.tfidf_extractor import TfidfExtractor
from src.models.baseline import BaselineModel
from src.training.evaluate import evaluate_model, print_evaluation_report

def main():
    print("Initializing components...")
    loader = DataLoader()
    cleaner = TextCleaner()
    extractor = TfidfExtractor(max_features=5000)
    model = BaselineModel(model_type='logistic_regression')

    # Paths
    data_file = "synthetic_news.csv"
    data_path = Path("data") / data_file
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # 1. Load Data
    if not data_path.exists():
        print("Data not found, creating synthetic dataset...")
        df = loader.create_synthetic_data(filename=data_file, num_samples=1000)
    else:
        print("Loading existing dataset...")
        df = loader.load_csv(data_file)

    # 2. Split Data
    print("Splitting dataset...")
    df_train, df_val, df_test = loader.get_train_val_test_split(df, text_col='text', label_col='label')

    # 3. Clean Text
    print("Cleaning text (this may take a moment)...")
    df_train['cleaned_text'] = df_train['text'].apply(cleaner.clean_text)
    df_val['cleaned_text'] = df_val['text'].apply(cleaner.clean_text)
    df_test['cleaned_text'] = df_test['text'].apply(cleaner.clean_text)

    # 4. Feature Extraction
    print("Extracting TF-IDF features...")
    X_train = extractor.fit_transform(df_train['cleaned_text'])
    X_val = extractor.transform(df_val['cleaned_text'])
    X_test = extractor.transform(df_test['cleaned_text'])

    y_train = df_train['label'].values
    y_val = df_val['label'].values
    y_test = df_test['label'].values

    # 5. Train Model
    print(f"Training {model.model_type} model...")
    model.fit(X_train, y_train)

    # 6. Evaluate Model on Validation Set
    print("Evaluating on Validation Set:")
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    
    val_metrics = evaluate_model(y_val, y_val_pred, y_val_prob)
    print(f"Validation Metrics: {val_metrics}")
    print_evaluation_report(y_val, y_val_pred)

    # 7. Evaluate Model on Test Set
    print("Evaluating on Test Set:")
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_prob)
    print(f"Test Metrics: {test_metrics}")

    # 8. Save Artifacts
    print("Saving model and vectorizer...")
    extractor.save(models_dir / "tfidf_vectorizer.pkl")
    model.save(models_dir / "baseline_model.pkl")
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
