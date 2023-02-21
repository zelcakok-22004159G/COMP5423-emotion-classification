import random
import numpy as np
import torch

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)

import pandas as pd
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, BertConfig
from torch.optim import AdamW

from libs.utils import split_tensor_datasets, get_training_dataset_loader, get_validate_dataset_loader
from libs.training_kit import TrainingKit
from libs.trainer import Trainer
from libs.data_preprocessor import DataProcessor

# Configs
epochs = 1
batch_size = 1
rows_per_batch = 1
columns = ["Sentence", "Emotion"]

# Prepare the datasets
train_df = pd.read_csv('data/train_data.txt', header=0, names=columns, sep=";")
val_df = pd.read_csv('data/val_data.txt', header=0, names=columns, sep=";")

train_df = pd.concat([train_df, val_df])
train_df = DataProcessor().process(train_df, columns)

# Init the training kit
training_kit = TrainingKit(
    train_df, 
    feat_col_name="Emotion", 
    data_col_name="Sentence", 
    row_size=batch_size * rows_per_batch,
)

train_df = training_kit.get_tensor_dataset()
train_ds, val_ds = split_tensor_datasets(train_df)

train_dataloader = get_training_dataset_loader(train_ds, batch_size=batch_size)
val_dataloader = get_validate_dataset_loader(val_ds, batch_size=batch_size)

# Prepare the model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=len(training_kit.features),  # The number of output labels.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)

optimizer = AdamW(model.parameters(),
                  lr=5e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  )

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

trainer = Trainer(model, optimizer, scheduler,
                  train_dataloader, val_dataloader, epochs, device="cpu")

trainer.train()

model.save_pretrained("model")
training_kit.save("model")
