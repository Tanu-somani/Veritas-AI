import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path

class BertClassifier:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)
        self.is_trained = False

    def predict(self, texts):
        self.model.eval()
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            
        return predictions.cpu().numpy(), probabilities.cpu().numpy()

    def save(self, directory: str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.is_trained = True

    def load(self, directory: str):
        self.model = DistilBertForSequenceClassification.from_pretrained(directory)
        self.tokenizer = DistilBertTokenizer.from_pretrained(directory)
        self.model.to(self.device)
        self.is_trained = True
