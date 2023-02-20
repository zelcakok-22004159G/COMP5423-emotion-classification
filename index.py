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

from utils import split_tensor_datasets, get_training_dataset_loader, get_validate_dataset_loader
from training_kit import TrainingKit
from trainer import Trainer

def debug_params(model):
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(
        len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# Configs
epochs = 3
batch_size = 5
rows_per_batch = 50

# Prepare the datasets
df = pd.read_csv('data/train_data.txt', header=0, names=["Sentence", "Emotion"], sep=";")

# Init the training kit
training_kit = TrainingKit(
    df, 
    feat_col_name="Emotion", 
    data_col_name="Sentence", 
    row_size=batch_size * rows_per_batch,
)

tensor_ds = training_kit.get_tensor_dataset()
train_ds, val_ds = split_tensor_datasets(tensor_ds, ratio=0.7)

train_dataloader = get_training_dataset_loader(train_ds, batch_size=batch_size)
val_dataloader = get_validate_dataset_loader(val_ds, batch_size=batch_size)

# Prepare the model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=len(training_kit.features),  # The number of output labels.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)
model.cpu()

optimizer = AdamW(model.parameters(),
                  lr=5e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  )

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

trainer = Trainer(model, optimizer, scheduler,
                  train_dataloader, val_dataloader, epochs)

trainer.train()

model.save_pretrained("model")
