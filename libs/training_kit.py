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

        samples, features = self.get_shuffled_ds(df, row_size)
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
    
    def get_shuffled_ds(self, df: pd.DataFrame, sampling_size: int):
        features = sorted(df[self.feat_col_name].unique())
        shuffled = np.random.permutation(df.to_numpy().tolist())
        return pd.DataFrame(shuffled, columns=[self.data_col_name, self.feat_col_name]), features

    def __compute(self):
        tokenizer = self.tokenizer
        input_ids, attention_masks = [], []
        for line in self.data:
            encoded_dict = tokenizer.encode_plus(
                ' '.join(line.split()[:181]),
                add_special_tokens=True,
                max_length=183,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
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