# <span style="color: #4CAF50;">NLP-Sentiment-Analysis-with-BERT</span>

<span style="color: #555555;">This project uses a BERT-based model to perform sentiment analysis on tweets related to games. The goal is to classify sentiments into three categories: <strong>Negative</strong>, <strong>Neutral</strong>, and <strong>Positive</strong>. The model is trained on a custom dataset containing tweets, with preprocessing, tokenization, and classification steps implemented.</span>

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Steps Involved](#steps-involved)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Advanced Analysis](#advanced-analysis)
- [Contributions](#contributions)

## Overview

This project applies Natural Language Processing (NLP) techniques to classify tweets related to games into **Negative**, **Neutral**, or **Positive** sentiments using the BERT transformer model. The workflow includes dataset loading, text preprocessing, tokenization, model training, and prediction, with results saved to a CSV file for further analysis.

## Installation

To install the required dependencies, run:

```bash
pip install transformers[torch] accelerate -U
```

## Usage
- Download and load the dataset (data.csv) containing tweets and sentiment labels.
- Preprocess the text by converting it to lowercase and handling missing values.
- Split the dataset into training and test sets (80% for training, 20% for testing).
- Load the BERT model and tokenizer.
- Tokenize and encode the text data.
- Train the model using Hugging Face's Trainer class with specified training parameters.
- Evaluate the model performance on the test dataset and predict sentiments.
- Save predictions to predicted_sentiments.csv.

## Steps Involved
- Data Preprocessing: Converts text to lowercase and handles missing values.
- Tokenization: Uses the BERT tokenizer to tokenize text data.
- Sentiment Label Encoding: Converts sentiment labels (Negative, Neutral, Positive) to numerical format.
- Model Training: Utilizes the BERT model with fine-tuning and training strategies.
- Prediction: The model predicts sentiment for the test set and stores results.

## Models
- BERT-based model for sequence classification: Uses the bert-base-uncased pre-trained model for sentiment analysis. The model is fine-tuned with a custom dataset of tweets.
- Model Output: The model produces sentiment predictions with labels (Negative, Neutral, Positive) and corresponding confidence scores.

## Evaluation Metrics
The model’s performance is evaluated using:

. Accuracy
. Precision
. Recall
. F1-Score
These metrics are displayed in a classification report after evaluating the model on the test set.

## Advanced Analysis
- Confusion Matrix: For visualizing the classification performance.
- Sentiment Distribution: Visualizations showing the distribution of sentiments in the dataset.
- Performance Metrics for Each Class: Analysis of recall and precision for each sentiment label (Negative, Neutral, Positive).

## Contributions
Feel free to contribute to this project by:

1. Reporting issues or bugs.
2. Proposing improvements.
3. Adding new features or enhancements.
