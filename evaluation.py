'''
    Filename: evaluation.py
    Usage: Standalone file to evaluate the model
'''
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

model_folder = "model"

features = ["anger", "fear", "joy", "love", "sadness", "surprise"]
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    model_folder,
    num_labels=len(features)
)

def classify(line):
    inputs = tokenizer(line, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits

    logits = logits.detach().cpu().numpy()
    [pred_flat] = np.argmax(logits, axis=1).flatten()
    return features[pred_flat]


df = pd.read_csv("data/test_data.txt", names=["id", "class"])

# Classify the line one by one
for i in tqdm(range(0, len(df))):
    line = df.at[i, 'id']
    df.at[i, 'class'] = classify(line)
    df.at[i, 'id'] = i

df.to_csv(f"submission-{model_folder}.csv", index=False, header=["id", "class"])
    