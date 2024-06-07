In this implementation, we use the pre-trained BERT model as the transformer backbone to generate sentence embeddings. The choices made regarding the model architecture outside of the transformer backbone are:

Pooling Strategy: We use the mean pooling of the output embeddings across the sequence dimension (dim=1) to obtain a single fixed-length embedding for each sentence. This is a common approach to obtain sentence representations from transformer models like BERT.
Rationale: Mean pooling is a simple and effective way to aggregate the sequence of output embeddings into a single vector representation while considering all token embeddings. It provides a good trade-off between computational complexity and performance for many downstream tasks.
Input Representation: We use the pre-trained BERT tokenizer to tokenize the input sentences and handle padding and truncation as needed.
Rationale: The BERT tokenizer is specifically designed for the BERT model and handles tokenization, padding, and truncation efficiently. Using the pre-trained tokenizer ensures that the input is properly processed and compatible with the pre-trained model weights.
Output Representation: The sentence embeddings are directly obtained from the mean-pooled output embeddings of the transformer model, without any additional transformation or projection.
Rationale: The output embeddings from the pre-trained BERT model are already highly informative and can be used directly for many downstream tasks without additional transformations. This simplifies the model architecture and reduces the number of trainable parameters.