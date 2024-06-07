# Task 1: Sentence Transformer Implementation
# Implement a sentence transformer model using any deep learning framework of your choice. 
# This model should be able to encode input sentences into fixed-length embeddings. 
# Test your implementation with a few sample sentences and showcase the obtained embeddings. 
# Describe any choices you had to make regarding the model architecture outside of the transformer backbone.






from transformers import AutoTokenizer, AutoModel
import torch

class SentenceTransformer:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def encode(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        # Use the mean of the output embeddings as the sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

# Test the implementation
if __name__ == "__main__":
    #sentences = ["Hello, how are you?", "I am fine, thank you!", "Transformers are powerful models for NLP."]
    sentences = ["how are you?", "I am fine, thank you!", "Today i am feeling happy", "Today i had a bad day", "Did you see the new movie?", "Did you heard the news of accident?"]
    model = SentenceTransformer()
    embeddings = model.encode(sentences)
    print(embeddings)

#