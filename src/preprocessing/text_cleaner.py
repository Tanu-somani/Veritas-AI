import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# We will download NLTK data lazily when needed
_nltk_downloaded = False

def _download_nltk_data():
    global _nltk_downloaded
    if not _nltk_downloaded:
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        _nltk_downloaded = True

class TextCleaner:
    def __init__(self, use_lemmatization: bool = True):
        _download_nltk_data()
        self.use_lemmatization = use_lemmatization
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text by applying lowercasing, removing punctuation, 
        stopwords, and optionally lemmatizing.
        """
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 3. Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # 4. Remove punctuation and numbers
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # 5. Tokenize (simple split by whitespace is often faster and sufficient after punctuation removal)
        tokens = text.split()
        
        # 6. Remove stopwords and lemmatize
        cleaned_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                if self.use_lemmatization:
                    token = self.lemmatizer.lemmatize(token)
                cleaned_tokens.append(token)
                
        return ' '.join(cleaned_tokens)

if __name__ == "__main__":
    cleaner = TextCleaner()
    sample = "The quick brown fox jumps over 123 the lazy dogs! https://example.com"
    print(f"Original: {sample}")
    print(f"Cleaned:  {cleaner.clean_text(sample)}")
