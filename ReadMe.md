# Sentence Transformer with Multi-Task Learning

## Introduction

This project implements a sentence transformer using PyTorch and the Hugging Face `transformers` library. The model is expanded to handle multi-task learning, including sentence classification and sentiment analysis. Additionally, layer-wise learning rates are implemented to enhance training.

## Setup

1. Clone the repository:

   ```sh
   git clone <repository-url>
   cd sentence_transformer_project

2. Create a virtual environment and activate it:

python3 -m venv venv
source venv/bin/activate

3. Install the required packages:

pip install -r requirements.txt

Task 1: Sentence Transformer Implementation
Run the sentence_transformer.py script to test the sentence transformer:
python sentence_transformer.py

Task 2: Multi-Task Learning Expansion
The script also includes a multi-task learning expansion. It demonstrates both sentence classification and sentiment analysis.