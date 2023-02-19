import torch
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import TensorDataset
import numpy as np
from gensim.parsing.preprocessing import remove_stopwords

class TrainingKit:
    def __init__(self, df: pd.DataFrame, feat_col_name: str, data_col_name: str, row_size=50, **kwargs):
        self.row_size = row_size
        self.feats = getattr(df, feat_col_name)[:row_size]
        self.data = getattr(df, data_col_name)[:row_size]
        self.features = set(self.feats)
        self.label2id = {label: id for id, label in enumerate(self.features)}
        self.labels = torch.tensor(
            [self.label2id[label] for label in self.feats]
        )
        self.options = kwargs
        self.__compute()

    @property
    def tokenizer(self):
        if self.options.get("tokenizer"):
            return self.options.get("tokenizer")
        return BertTokenizer.from_pretrained('bert-base-uncased')
    
    def process_line(self, line):
        return ' '.join(np.random.permutation(remove_stopwords(line).split())[:100])
    
    def __compute(self):
        tokenizer = self.tokenizer.from_pretrained('bert-base-uncased')
        input_ids, attention_masks = [], []
        for line in self.data:
            encoded_dict = tokenizer.encode_plus(
                self.process_line(line),
                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                max_length = 102,           # Pad & truncate all sentences.
                pad_to_max_length = True,
                return_attention_mask = True,   # Construct attn. masks.
                return_tensors = 'pt',     # Return pytorch tensors.
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def get_tensor_dataset(self):
        return TensorDataset(self.input_ids, self.attention_masks, self.labels)
