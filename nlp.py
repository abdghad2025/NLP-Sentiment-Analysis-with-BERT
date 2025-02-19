"""NLP.ipynb
"""

!pip install transformers[torch] accelerate -U

"""#First, we import the necessary libraries and modules.
#Pandas: for working with data.
#Sklearn: for data segmentation and model evaluation.
#Transformers: for loading the BERT model and tokenizer and for training and prediction.
#torch: for GPU support and tensor management.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import TextClassificationPipeline
import torch
from torch.utils.data import Dataset, DataLoader

"""#This block checks if a GPU is available and sets the device accordingly."""

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using the CPU instead.")

"""#data.csv: dataset containing tweets and their associated emotional tags.
#header=None, names=['ID', 'Game', 'Sentiment', 'Text']: Specifies that the dataset has no header and provides column names.

"""

# Load dataset
data = pd.read_csv('data.csv', header=None, names=['ID', 'Game', 'Sentiment', 'Text'])

"""#Converts text to lowercase and handles missing values by replacing them with an empty string."""

# Load dataset
data = pd.read_csv('data.csv', header=None, names=['ID', 'Game', 'Sentiment', 'Text'])

# Preprocess the text data
def preprocess_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower()

data['Text'] = data['Text'].apply(preprocess_text)

"""#It divides the data into training (80%) and test (20%) sets."""

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

"""#Loads the tokenizer and BERT model with three output labels (negative, neutral, positive).
#If available, it transfers the model to the GPU.
"""

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.to(device)

"""#Tokenizes, truncates, or tokens text data up to 512 tokens long."""

# Tokenize the text data
train_encodings = tokenizer(train_data['Text'].tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_data['Text'].tolist(), truncation=True, padding=True, max_length=512)

"""#Converts sentiment labels (negative, neutral, positive) to numeric values (0, 1, 2)."""

# Convert sentiment labels to numerical format
def sentiment_to_label(sentiment):
    if sentiment == 'Negative':
        return 0
    elif sentiment == 'Neutral':
        return 1
    else:
        return 2

train_labels = train_data['Sentiment'].apply(sentiment_to_label).tolist()
test_labels = test_data['Sentiment'].apply(sentiment_to_label).tolist()

"""#Defines a custom data collection class that returns marked entries and labels."""

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

"""#It specifies training parameters such as number of courses, batch size and logging."""

# Define training arguments with optimization
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=2  # Accumulate gradients to effectively increase batch size
)

"""#Initializes the Trainer with the model, training arguments, and dataset."""

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

"""#It trains the model."""

# Train the model
trainer.train()

"""#Evaluates the model on the test set."""

# Evaluate the model
eval_result = trainer.evaluate()

"""#It predicts sentiment labels for the test set and extracts the predicted label with the highest score."""

# Predict sentiment for the test set
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)

"""#Prints a classification report showing the precision, recall, and F1 score for each emotion class."""

# Print classification report
print(classification_report(test_labels, preds, target_names=['Negative', 'Neutral', 'Positive']))

"""#Creates a pipeline for sentiment analysis using a trained model."""

# Create a sentiment analysis pipeline
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=0 if torch.cuda.is_available() else -1)

"""#The pipeline applies sentiment analysis to the original dataset and stores the predictions."""

# Predict sentiment for the original dataset
data['Predicted_Sentiment'] = data['Text'].apply(lambda x: pipeline(x)[0])

"""#Saves results to a CSV file, including predicted sentiment."""

# Save the results
data.to_csv('predicted_sentiments.csv', index=False)

"""#The model has been successfully trained.

Training results:
Global Step: 7468
Training Loss: 0.4083
Train Runtime: 2105.7764 seconds
Train Samples per Second: 56.744
Train Steps per Second: 3.546
Total FLOPS: 1.9341*〖10〗^16
Epochs: 2.0

This model has performed very well with an overall accuracy of 92%.

Detailed performance metrics:
Negative emotions:
Accuracy: 0.95
Recall: 0.91
F1 score: 0.93
Support: 4519

Neutral emotions:
Accuracy: 0.93
Recall: 0.88
F1 score: 0.90
Support: 3596

Positive emotions:
Accuracy: 0.90
Recall: 0.95
F1 score: 0.93
Support: 6822

General criteria:
Accuracy: 0.92

Macro Average:
Accuracy: 0.93
Recall: 0.91
F1 score: 0.92

Weighted Average:
Accuracy: 0.92
Recall: 0.92
F1 score: 0.92
This model has high accuracy and recall for negative and positive emotions, which shows that it is good at accurately identifying these emotions.

The predicted_sentiments.csv file contains the original data along with model predictions for sentiment analysis.

ID: A unique identifier for each record.
Game: Game name mentioned in the text.
Sentiment: The actual sentiment label assigned to the text (eg, positive, negative, neutral).
Text: The text content of the tweet or comment.
Predicted_Sentiment: model prediction for sentiment in a detailed format, showing scores for each label (negative, neutral, positive).

Example of predictions in CSV file:
ID,Game,Sentiment,Text,Predicted_Sentiment
2401,Borderlands,Positive,"im getting on borderlands and i will murder you all ,","[{'label': 'LABEL_0', 'score': 0.00022317680122796446}, {'label': 'LABEL_1', 'score': 0.00030716147739440203}, {'label': 'LABEL_2', 'score': 0.9994696974754333}]"
2401,Borderlands,Positive,"i am coming to the borders and i will kill you all,","[{'label': 'LABEL_0', 'score': 0.00018267231644131243}, {'label': 'LABEL_1', 'score': 0.0004388584347907454}, {'label': 'LABEL_2', 'score': 0.9993784427642822}]"
2401,Borderlands,Positive,"im getting on borderlands and i will kill you all,","[{'label': 'LABEL_0', 'score': 0.00019511835125740618}, {'label': 'LABEL_1', 'score': 0.0003024300967808813}, {'label': 'LABEL_2', 'score': 0.999502420425415}]"
2401,Borderlands,Positive,"im coming on borderlands and i will murder you all,","[{'label': 'LABEL_0', 'score': 0.00022581385564990342}, {'label': 'LABEL_1', 'score': 0.00030427431920543313}, {'label': 'LABEL_2', 'score': 0.9994699358940125}]"
2401,Borderlands,Positive,"im getting on borderlands 2 and i will murder you me all,","[{'label': 'LABEL_0', 'score': 0.0001595183421159163}, {'label': 'LABEL_1', 'score': 0.0003770108160097152}, {'label': 'LABEL_2', 'score': 0.9994634985923767}]"
2401,Borderlands,Positive,"im getting into borderlands and i can murder you all,","[{'label': 'LABEL_0', 'score': 0.0002854418125934899}, {'label': 'LABEL_1', 'score': 0.000294297729851678}, {'label': 'LABEL_2', 'score': 0.9994202852249146}]"
2402,Borderlands,Positive,so i spent a few hours making something for fun. . . if you don't know i am a huge @borderlands fan and maya is one of my favorite characters. so i decided to make myself a wallpaper for my pc. . here is the original image versus the creation i made :) enjoy! pic.twitter.com/mlsi5wf9jg,"[{'label': 'LABEL_0', 'score': 0.0001660394627833739}, {'label': 'LABEL_1', 'score': 0.0003505792119540274}, {'label': 'LABEL_2', 'score': 0.9994833469390869}]"
2402,Borderlands,Positive,"so i spent a couple of hours doing something for fun... if you don't know that i'm a huge @ borderlands fan and maya is one of my favorite characters, i decided to make a wallpaper for my pc.. here's the original picture compared to the creation i made:) have fun! pic.twitter.com / mlsi5wf9jg","[{'label': 'LABEL_0', 'score': 0.00017550277698319405}, {'label': 'LABEL_1', 'score': 0.0003316322108730674}, {'label': 'LABEL_2', 'score': 0.999492883682251}]"
2402,Borderlands,Positive,so i spent a few hours doing something for fun... if you don't know i'm a huge @ borderlands fan and maya is one of my favorite characters.,"[{'label': 'LABEL_0', 'score': 0.0001786408683983609}, {'label': 'LABEL_1', 'score': 0.0003270627639722079}, {'label': 'LABEL_2', 'score': 0.9994943141937256}]"
Each entry in the Predicted_Sentiment column is a JSON-like string containing a list of dictionaries. Each dictionary represents a label (sense category) with its corresponding score. Tags are:
LABEL_0: negative
LABEL_1: Neutral
LABEL_2: Positive
Model confidence scores for each label. The label with the highest score is the predicted sentiment for that text.

For example, in the first row:
"[{'label': 'LABEL_0', 'score': 0.00022317680122796446}, {'label': 'LABEL_1', 'score': 0.00030716147739440203}, {'label': 'LABEL_2', 'score': 0.9994696974754333}]"
LABEL_0 (negative): score = 0.00022317680122796446
LABEL_1 (neutral): score = 0.00030716147739440203
LABEL_2 (positive): score = 0.9994696974754333
Since LABEL_2 has the highest score, the model predicts sentiment positively.

The CSV file provides a detailed view of the model's sentiment prediction for each text input. This format is useful for analyzing the model's performance and understanding its confidence in its predictions in different emotions.

"""