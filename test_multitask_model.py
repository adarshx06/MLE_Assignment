# test_multitask_model.py

import torch
from transformers import AutoTokenizer
from models.multitask_model import MultiTaskSentenceTransformer

# Initialize model and tokenizer
model_name = 'bert-base-uncased'
model = MultiTaskSentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample sentences
#sentences = ["Hello, how are you?", "I am fine, thank you!", "Transformers are powerful models for NLP.","Today i am happy", "Its a lousy day"]
sentences = ["how are you?", "I am fine, thank you!", "Today i am feeling happy", "Today i had a bad day", "Did you see the new movie?", "Did you heard the news of accident?"]
# Tokenize sentences
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128)

# Forward pass
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
classification_logits, sentiment_logits = model(input_ids, attention_mask)

# Print the results
print("Classification logits:")
print(classification_logits)
print("\nSentiment logits:")
print(sentiment_logits)

# Check dimensions
assert classification_logits.shape == (len(sentences), 3), "Classification logits shape is incorrect."
assert sentiment_logits.shape == (len(sentences), 3), "Sentiment logits shape is incorrect."

print("\nTest passed successfully!")
