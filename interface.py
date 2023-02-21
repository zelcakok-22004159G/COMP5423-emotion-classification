from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

# Configs
columns = ["Sentence", "Emotion"]

# Prepare features
features = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
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
        input_ids, attention_masks = [], []
        encoded_dict = tokenizer.encode_plus(
            cls.process_line(line),
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=502,           # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,   # Construct attn. masks.
            return_tensors='pt',     # Return pytorch tensors.
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        with torch.no_grad():
            outputs = model(input_ids,
                            token_type_ids=None,
                            attention_mask=attention_masks)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            [pred_flat] = np.argmax(logits, axis=1).flatten()
            return features[pred_flat]

app = Flask(__name__)
CORS(app)

@app.route("/")
def main():
    return jsonify(status="ready") 

@app.route("/classify")
def classify():
    args = request.args
    q = args.get("q")
    if not q:
        return jsonify(result="", query="")
    return jsonify(result=Interface.classify(q), query=q)

if __name__ == "__main__":
    app.run()