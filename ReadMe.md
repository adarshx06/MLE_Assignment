# Sentence Transformer with Multi-Task Learning

## Introduction

This project implements a sentence transformer using PyTorch and the Hugging Face `transformers` library. The model is expanded to handle multi-task learning, including sentence classification and sentiment analysis. Additionally, layer-wise learning rates are implemented to enhance training.

## Setup

1. Create a virtual environment and activate it:

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

3. Task 1: Sentence Transformer Implementation
    ```sh
    Run the sentence_transformer.py script to test the sentence transformer:
    python sentence_transformer.py
    ```

4. Task 2: Multi-Task Learning Expansion
The multitask_model.py includes a multi-task learning expansion. It demonstrates both sentence classification and sentiment analysis.

5. Task 3 and Task 4:
You can see the MLE_Fetch Notebook for Task 3 and Task 4.

6. Docker Container
To package the project in a Docker container:
Create a Dockerfile:

    ```sh
    FROM python:3.8-slim
    WORKDIR /app
    COPY requirements.txt requirements.txt
    RUN pip install -r requirements.txt
    COPY . .
    CMD ["python", "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    ```

7. Build the Docker image:

    ```sh
    docker build -t mleassigment
    ```

8. Run the Docker container:

    ```sh
    docker run -it --rm mleassigment
    ```

Thank you!
