# train.py

from transformers import AdamW, AutoTokenizer
import torch.nn as nn
from data.dummy_data import get_dataloader
from models.multitask_model import MultiTaskSentenceTransformer

def train_multitask_model(model, train_dataloader, epochs=3):
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        for batch in train_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels_classification = batch['labels_classification']
            labels_sentiment = batch['labels_sentiment']
            
            optimizer.zero_grad()
            classification_logits, sentiment_logits = model(input_ids, attention_mask)
            
            loss_classification = nn.CrossEntropyLoss()(classification_logits, labels_classification)
            loss_sentiment = nn.CrossEntropyLoss()(sentiment_logits, labels_sentiment)
            loss = loss_classification + loss_sentiment
            
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    model_name = 'bert-base-uncased'
    model = MultiTaskSentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataloader = get_dataloader(tokenizer)
    
    train_multitask_model(model, dataloader)
