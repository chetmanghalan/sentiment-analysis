# Sentiment Analysis of Movie Reviews Based on NLP Model

This project implements a sentiment analysis and topic modeling system using BERT, TextCNN, Focal Loss, LDA, and Word2Vec. The system is designed to analyze movie reviews, classify sentiments, and identify underlying themes within the reviews.

## Project Structure

- `main.py`: Main script to run the sentiment analysis model training and evaluation.
- `topic_modeling.py`: Script to run the topic modeling and sentiment analysis using LDA and Word2Vec.
- `SentimentDataset`: Custom dataset class to handle text data and labels.
- `SentimentModel`: Custom model class combining BERT and TextCNN.
- `FocalLoss`: Custom loss function to address class imbalance.
- `requirements.txt`: List of required packages.

## Setup

### Prerequisites

- Python 3.6+
- PyTorch
- Transformers
- pandas
- scikit-learn
- gensim
- jieba

### Installation

  1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Dataset

The dataset should be an Excel file:
- `text`: The movie review text.
- `label`: The sentiment label associated with the review.

## How to Run

### Sentiment Analysis

1. Ensure your dataset is stored in the specified location and has the correct format.
2. Run the `main.py` script to train and evaluate the sentiment analysis model:
    ```bash
    python main.py
    ```

### Topic Modeling and Sentiment Analysis

1. Ensure your dataset is stored in the specified location and has the correct format.
2. Run the `topic_modeling.py` script to perform topic modeling and sentiment analysis:
    ```bash
    python topic_modeling.py
    ```

## Script Details

### Sentiment Analysis (`main.py`)

- **Data Preparation**
  - Load the dataset from an Excel file.
  - Split the data into training and testing sets.
  - Tokenize the text using BERT tokenizer.

- **Model Architecture**
  - BERT for context comprehension.
  - TextCNN for extracting local features.
  - Focal Loss to handle class imbalance.

- **Training and Evaluation**
  - Train the model using AdamW optimizer and learning rate scheduler.
  - Evaluate the model on the test set and save the best model.

### Topic Modeling and Sentiment Analysis (`topic_modeling.py`)

- **Data Preparation**
  - Load the dataset from an Excel file.
  - Tokenize the reviews using `jieba`.

- **Topic Modeling**
  - Use `CountVectorizer` to convert text data into a matrix of token counts.
  - Fit an LDA model to the token count matrix to extract topics.
  - Train a Word2Vec model on the tokenized reviews to find similar words for initial keywords.
  - Combine topic keywords, custom attributes, and initial keywords to form the attribute set for each topic.

- **Topic Classification**
  - Classify reviews into topics based on the presence of attributes.
  - Merge topic classification results with the original dataset to get sentiment labels.
  - Calculate and display the label percentage for each topic.

## Results

### Sentiment Analysis

The script outputs the training and validation accuracy and loss for each epoch. The best model is saved based on the validation accuracy.

### Topic Modeling and Sentiment Analysis

The script outputs the percentage of each sentiment label for each topic, providing insights into the distribution of sentiments across different topics.

