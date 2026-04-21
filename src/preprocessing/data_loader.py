import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import os

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load dataset from a CSV file."""
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return pd.read_csv(file_path)
    
    def get_train_val_test_split(self, df: pd.DataFrame, text_col: str, label_col: str, 
                                 test_size: float = 0.2, val_size: float = 0.1, 
                                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train, validation, and test sets."""
        from sklearn.model_selection import train_test_split
        
        # First split into train_val and test
        df_train_val, df_test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df[label_col]
        )
        
        # Adjust val_size to be a proportion of train_val
        val_prop = val_size / (1 - test_size)
        
        # Then split train_val into train and val
        df_train, df_val = train_test_split(
            df_train_val, test_size=val_prop, random_state=random_state, stratify=df_train_val[label_col]
        )
        
        return df_train, df_val, df_test

    def create_synthetic_data(self, filename: str = "synthetic_news.csv", num_samples: int = 1000):
        """Creates a synthetic dataset for testing if no real dataset is provided."""
        real_news = [
            "The stock market saw a significant increase today following the Federal Reserve's announcement.",
            "Scientists have discovered a new exoplanet that could potentially support water.",
            "The local government passed a new bill to improve public transportation infrastructure.",
            "A major tech company unveiled its latest smartphone featuring an advanced AI chip.",
            "The upcoming election is seeing record voter registration numbers across the state."
        ] * (num_samples // 10)
        
        fake_news = [
            "Aliens have landed in Central Park and are distributing free Wi-Fi routers.",
            "Eating five pounds of chocolate a day is scientifically proven to reverse aging completely.",
            "A local man claims he built a time machine out of a microwave and a toaster.",
            "The earth is actually flat and NASA has been hiding the truth with CGI.",
            "Drinking bleach cures all known viruses, says anonymous online doctor."
        ] * (num_samples // 10)
        
        # Add slight variations to avoid exact duplicates
        import random
        real_news = [f"{text} {random.randint(1,1000)}" for text in real_news]
        fake_news = [f"{text} {random.randint(1,1000)}" for text in fake_news]
        
        texts = real_news + fake_news
        labels = [1] * len(real_news) + [0] * len(fake_news) # 1 for REAL, 0 for FAKE
        
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save
        file_path = self.data_dir / filename
        df.to_csv(file_path, index=False)
        print(f"Synthetic dataset created at {file_path}")
        return df

if __name__ == '__main__':
    # Initialize and create synthetic data
    loader = DataLoader()
    loader.create_synthetic_data()
