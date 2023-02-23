from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

# Configs
columns = ["Sentence", "Emotion"]

# Prepare features
features = ["anger", "fear", "joy", "love", "sadness", "surprise"]
label2id = {label: id for id, label in enumerate(features)}
labels = torch.tensor(
    [label2id[label] for label in features]
)

# Prepare the model
model = BertForSequenceClassification.from_pretrained("model")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class Interface:
    @classmethod
    def process_line(cls, line):
        return ' '.join(np.random.permutation(line.split())[:500])
    
    @classmethod
    def classify(cls, line):
        inputs = tokenizer(line, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        logits = logits.detach().cpu().numpy()
        [pred_flat] = np.argmax(logits, axis=1).flatten()
        return features[pred_flat]

app = Flask(__name__)
CORS(app)

@app.route("/")
def main():
    return jsonify(status="ready") 

@app.route("/api/classify")
def classify():
    args = request.args
    q = args.get("q")
    if not q:
        return jsonify(result="", query="")
    return jsonify(result=Interface.classify(q), query=q)

if __name__ == "__main__":
    app.run()