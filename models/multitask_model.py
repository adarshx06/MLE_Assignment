# models/multitask_model.py

from transformers import AutoModel
import torch.nn as nn

print("Importing MultiTaskSentenceTransformer...")

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=3, num_sentiment_classes=3):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classification_head = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.sentiment_head = nn.Linear(self.bert.config.hidden_size, num_sentiment_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        classification_logits = self.classification_head(pooled_output)
        sentiment_logits = self.sentiment_head(pooled_output)
        return classification_logits, sentiment_logits
