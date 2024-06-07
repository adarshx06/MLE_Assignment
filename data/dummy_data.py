# data/dummy_data.py

from torch.utils.data import DataLoader, Dataset
import torch

class DummyDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels_classification, labels_sentiment):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.labels_classification = labels_classification
        self.labels_sentiment = labels_sentiment
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer(self.sentences[idx], return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels_classification': torch.tensor(self.labels_classification[idx], dtype=torch.long),
            'labels_sentiment': torch.tensor(self.labels_sentiment[idx], dtype=torch.long)
        }

def get_dataloader(tokenizer):
    sentences = ["how are you?", "I am fine, thank you!", "Today i am feeling happy", "Today i had a bad day", "Did you see the new movie?", "Did you heard the news of accident?"]
    labels_classification = [0, 1, 1, 2, 0, 2]  # Example classification labels
    labels_sentiment = [0, 1, 1, 2, 0, 2]  # Example sentiment labels (0: neutral, 1: positive, 2: negative)
    
    dataset = DummyDataset(tokenizer, sentences, labels_classification, labels_sentiment)
    return DataLoader(dataset, batch_size=2)
