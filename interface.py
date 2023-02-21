import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification


from libs.data_preprocessor import DataProcessor
from libs.utils import showPrediction

def process_line(line):
    return ' '.join(np.random.permutation(line.split())[:500])

features = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']

label2id = {label: id for id, label in enumerate(features)}
labels = torch.tensor(
    [label2id[label] for label in features]
)

model = BertForSequenceClassification.from_pretrained('model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

line = "The sky is beautful, isn't it?"
line = DataProcessor().process_words(line)

input_ids, attention_masks = [], []
encoded_dict = tokenizer.encode_plus(
    process_line(line),
    add_special_tokens=True,
    max_length=502,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',
)
input_ids.append(encoded_dict['input_ids'])
attention_masks.append(encoded_dict['attention_mask'])
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

with torch.no_grad():
    outputs = model(
        input_ids, 
        token_type_ids=None, 
        attention_mask=attention_masks, 
        labels=labels
    )

