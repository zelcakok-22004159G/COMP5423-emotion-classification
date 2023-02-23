import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification


features = ["anger", "fear", "joy", "love", "sadness", "surprise"]
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "model-v3",
    num_labels=len(features)
)
weights = torch.FloatTensor([0.8648, 0.8806, 0.6630, 0.9177, 0.7103, 0.9637]).cpu()
criterion = torch.nn.CrossEntropyLoss(weight=weights,reduction='mean')

def classify(line):
    inputs = tokenizer(line, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits

    logits = logits.detach().cpu().numpy()
    [pred_flat] = np.argmax(logits, axis=1).flatten()
    return features[pred_flat]


df = pd.read_csv("data/test_data_trimmed.txt", names=["id", "class"])

for i in range(0, len(df)):
    line = df.at[i, 'id']
    df.at[i, 'class'] = classify(line)
    df.at[i, 'id'] = i
    print("Processed", line)

df.to_csv("submission_model_v3.csv", index=False, header=["id", "class"])
    