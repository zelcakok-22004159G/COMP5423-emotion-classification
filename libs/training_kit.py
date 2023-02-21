import numpy as np
import pandas as pd
import torch
from json import dumps

from torch.utils.data import TensorDataset
from transformers import BertTokenizer


class TrainingKit:
    def __init__(self, df: pd.DataFrame, feat_col_name: str, data_col_name: str, row_size=50, **kwargs):
        self.feat_col_name = feat_col_name
        self.data_col_name = data_col_name

        samples, features = self.__random_sampling(df, row_size)
        self.feats = getattr(samples, feat_col_name)
        self.data = getattr(samples, data_col_name)

        self.features = features
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

    def shuffle_rows(self, line):
        return ' '.join(np.random.permutation(line.split())[:500])

    def __random_sampling(self, df: pd.DataFrame, sampling_size: int):
        buff = {}
        features = sorted(df[self.feat_col_name].unique())
        for row in df.to_numpy().tolist():
            [_, feat] = row
            if not buff.get(feat):
                buff[feat] = []
            if len(buff[feat]) >= sampling_size:
                continue
            buff[feat].append(row)

        samples = []
        for feat in features:
            for row in buff[feat]:
                samples.append(row)
        samples = np.random.permutation(samples)
        return pd.DataFrame(samples, columns=[self.data_col_name, self.feat_col_name]), features

    def __compute(self):
        tokenizer = self.tokenizer.from_pretrained('bert-base-uncased')
        input_ids, attention_masks = [], []
        for line in self.data:
            encoded_dict = tokenizer.encode_plus(
                self.shuffle_rows(line),
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
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def get_tensor_dataset(self):
        return TensorDataset(self.input_ids, self.attention_masks, self.labels)
    
    def save(self, folder):
        config = {
            "features": self.features,
            "label2id": self.label2id
        }
        with open(f"{folder}/training-kit.config", "w") as f:
            f.write(dumps(config, indent=4))
